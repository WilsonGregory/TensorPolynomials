import sys
import time
import numpy as np
from typing_extensions import Optional, Self
import wandb
import argparse
import itertools as it
import math

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx
import lineax as lx

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils
import tensorpolynomials.data as data

import signax


def get_data(
    D: int, n_train: int, n_val: int, n_test: int, sig_order: int, steps: int, key: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    integrator_steps = 1000
    subkey1, subkey2, subkey3 = random.split(key, num=3)

    train_X, train_Y = data.get_signature(D, n_train, integrator_steps, sig_order, steps, subkey1)
    val_X, val_Y = data.get_signature(D, n_val, integrator_steps, sig_order, steps, subkey2)
    test_X, test_Y = data.get_signature(D, n_test, integrator_steps, sig_order, steps, subkey3)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


def count_params(model) -> int:
    return sum(
        [
            0 if x is None else x.size
            for x in eqx.filter(jax.tree_util.tree_leaves(model), eqx.is_array)
        ]
    )


class SignaxOnly(eqx.Module):
    max_k: int

    def __init__(self: Self, max_k: int):
        self.max_k = max_k

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return signax.signature(x[None], self.max_k)[0]


class Baseline(eqx.Module):

    net: eqx.nn.MLP

    max_k: int

    def __init__(
        self: Self, D: int, max_k: int, steps: int, width: int, depth: int, key: jax.Array
    ) -> None:
        self.max_k = max_k
        n_in = steps * D
        n_out = sum([D**k for k in range(1, 1 + max_k)])

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.net(x.reshape(-1))


class EquivSignature(eqx.Module):

    net: eqx.nn.MLP

    max_k: int
    bilipschitz: bool
    eps: float

    def __init__(
        self: Self,
        max_k: int,
        steps: int,
        width: int,
        depth: int,
        bilipschitz: bool,
        key: jax.Array,
        eps: float = 1e-4,
    ) -> None:
        self.max_k = max_k
        self.bilipschitz = bilipschitz
        self.eps = eps

        n_in = steps + steps * (steps - 1) // 2
        n_out = sum([steps**k for k in range(1, 1 + max_k)])
        n_out += utils.metric_tensor_basis_size(max_k, steps)

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        n, D = x.shape
        X = x @ x.T

        if self.bilipschitz:
            eigvals, eigvecs = jnp.linalg.eigh(X, symmetrize_input=False)
            X = eigvecs @ jnp.diag(jnp.sqrt(eigvals + self.eps)) @ eigvecs.T
            # eps should be at least 1e-4 to avoid numerical issues

        X = X[jnp.triu_indices(n)].reshape(-1)

        assert X.shape == (n + n * (n - 1) // 2,)

        all_coeffs = self.net(X)

        out = jnp.zeros(0)
        idx = 0
        kth_basis = {0: jnp.ones(1)}  # used for k >=2 when kronecker delta gets involved
        for k in range(1, 1 + self.max_k):
            ein_str = ",".join([f"{utils.LETTERS[i] + utils.LETTERS[i+13]}" for i in range(k)])
            ein_str += f"->{utils.LETTERS[:k] + utils.LETTERS[13:13+k]}"
            tensor_inputs = (x,) * k
            basis = jnp.einsum(ein_str, *tensor_inputs).reshape((n**k,) + (D,) * k)
            kth_basis[k] = basis

            for j in range(k // 2):
                n_kron_deltas = j + 1
                remaining_k = k - n_kron_deltas * 2

                # first get tensor product of n_kron_deltas
                ein_str = ",".join([f"{utils.LETTERS[2*i:2*i+2]}" for i in range(n_kron_deltas)])
                tensor_inputs = (jnp.eye(D),) * n_kron_deltas
                kron_delta_product = jnp.einsum(ein_str, *tensor_inputs)
                # I might want to store this
                permuted_kron_deltas = jnp.stack(
                    [
                        kron_delta_product.transpose(idxs)
                        for idxs in utils.metric_tensor_r(n_kron_deltas * 2)
                    ]
                )

                kron_delta_basis = jnp.einsum(
                    f"Y{utils.LETTERS[:remaining_k]},Z{utils.LETTERS[remaining_k:k]}->YZ{utils.LETTERS[:k]}",
                    kth_basis[remaining_k],
                    permuted_kron_deltas,
                ).reshape((-1,) + (D,) * k)

                # do the final permutations, mixing the remaining_k axes with the kron_delta axes
                kron_delta_basis = jnp.stack(
                    [
                        kron_delta_basis.transpose(idxs)
                        for idxs in utils.final_permutations(k, remaining_k, 1)
                    ]
                ).reshape((-1,) + (D,) * k)
                basis = jnp.concatenate([basis, kron_delta_basis])

            collapsed_basis = basis.reshape((len(basis), D**k))
            coeffs = all_coeffs[idx : idx + len(basis)]
            out = jnp.concatenate([out, coeffs @ collapsed_basis])
            idx += len(basis)

        assert idx == len(all_coeffs)

        return out


def map_and_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """

    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
    """
    batch, _, D = x.shape
    assert callable(model)
    pred_signature = jax.vmap(model)(x)

    idx = 0
    k = 1
    loss_per_tensor = jnp.zeros((batch, 0))
    while idx < y.shape[1]:
        pred_flat_tensor = pred_signature[:, idx : idx + D**k]
        y_flat_tensor = y[:, idx : idx + D**k]
        loss_per_tensor = jnp.concatenate(
            [
                loss_per_tensor,
                jnp.mean((pred_flat_tensor - y_flat_tensor) ** 2, axis=1, keepdims=True),
            ],
            axis=1,
        )
        idx += D**k
        k += 1

    assert idx == y.shape[1]
    # mean over batch and tensor types. And previously we had mean over tensor entries, so there
    # error in the 1-tensor is balanced with the error in the 2-tensor.
    return jnp.mean(loss_per_tensor), aux_data


def map_and_loss_return_map(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
) -> tuple[jax.Array, eqx.nn.State | None, jax.Array]:
    """
    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
    """
    batch, _, D = x.shape
    assert callable(model)
    pred_signature = jax.vmap(model)(x)

    idx = 0
    k = 1
    loss_per_tensor = jnp.zeros((batch, 0))
    while idx < y.shape[1]:
        pred_flat_tensor = pred_signature[:, idx : idx + D**k]
        y_flat_tensor = y[:, idx : idx + D**k]
        loss_per_tensor = jnp.concatenate(
            [
                loss_per_tensor,
                jnp.mean((pred_flat_tensor - y_flat_tensor) ** 2, axis=1, keepdims=True),
            ],
            axis=1,
        )
        idx += D**k
        k += 1

    assert idx == y.shape[1]
    # mean over batch and tensor types. And previously we had mean over tensor entries, so there
    # error in the 1-tensor is balanced with the error in the 2-tensor.
    return jnp.mean(loss_per_tensor), aux_data, pred_signature


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("--batch", help="batch size", type=int, default=32)
    parser.add_argument("--n-train", help="number of training points", type=int, default=1024)
    parser.add_argument("--n-val", help="number of validation points", type=int, default=1024)
    parser.add_argument("--n-test", help="number of testing points", type=int, default=1024)
    parser.add_argument("-t", "--n-trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("--seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save-model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load-model", help="file name to load params from", type=str, default=None
    )
    parser.add_argument(
        "-v", "--verbose", help="verbose argument passed to trainer", type=int, default=1
    )
    parser.add_argument(
        "--normalize",
        help="normalize input data, equivariantly",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--sig-order", help="max order of the tensor signature", type=int, default=3
    )
    parser.add_argument(
        "--wandb",
        help="whether to use wandb on this run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-entity", help="the wandb user", type=str, default="wilson_gregory")
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="signature-tensors-2"
    )

    return parser.parse_args()


# MAIN
args = handleArgs(sys.argv)
D = 3
steps = 10

key = random.PRNGKey(time.time_ns() if args.seed is None else args.seed)
train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(
    D, args.n_train, args.n_val, args.n_test, args.sig_order, steps, key
)

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    ("signax_only", 1, True, SignaxOnly(args.sig_order)),
    ("tensor_poly", 5e-4, True, EquivSignature(args.sig_order, steps, 32, 3, False, subkey1)),
    # ("tensor_poly_bilipschitz", 5e-4, True, EquivSignature(sig_order, steps, 32, 3, True, subkey2)),
    ("baseline_samewidth", 5e-3, True, Baseline(D, args.sig_order, steps, 32, 3, subkey4)),
    ("baseline_sameparams", 1e-3, True, Baseline(D, args.sig_order, steps, 128, 3, subkey3)),
]

results = np.zeros((args.n_trials, len(models_list), 3))
for t in range(args.n_trials):
    for k, (model_name, lr, needs_training, model) in enumerate(models_list):

        print(f"{model_name}: {count_params(model):,} params")

        name = f"signature_{model_name}_t{t}_lr{lr}_e{args.epochs}_ntrain{args.n_train}"
        save_model = f"{args.save_model}/{name}.eqx" if args.save_model is not None else None

        if needs_training:
            if args.load_model:
                trained_model = ml.load(f"{args.load_model}/{name}.eqx", model)
            else:
                if args.wandb:
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,  # what is this?
                        name=name,
                        settings=wandb.Settings(start_method="fork"),
                    )
                    wandb.config.update(args)
                    wandb.config.update(
                        {
                            "trial": t,
                            "model_name": model_name,
                            "lr": lr,
                        }
                    )

                steps_per_epoch = int(args.n_train / args.batch)
                key, subkey = random.split(key)
                trained_model, _, _, _ = ml.train(
                    train_X,
                    train_Y,
                    map_and_loss,
                    model,
                    subkey,
                    ml.EpochStop(epochs=args.epochs, verbose=args.verbose),
                    args.batch,
                    optax.adamw(
                        optax.cosine_onecycle_schedule(
                            args.epochs * steps_per_epoch, lr, div_factor=3
                        )
                    ),
                    validation_X=val_X,
                    validation_Y=val_Y,
                    save_model=save_model,
                    is_wandb=args.wandb,
                )

                if args.wandb:
                    wandb.finish()

                if args.save_model is not None:
                    ml.save(save_model, trained_model)
        else:
            trained_model = model

        key, subkey1, subkey2, subkey3 = random.split(key, num=4)
        results[t, k, 0] = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            train_X,
            train_Y,
            args.batch,
            subkey1,
            devices=jax.devices(),
        )
        results[t, k, 1] = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            val_X,
            val_Y,
            args.batch,
            subkey1,
            devices=jax.devices(),
        )
        test_loss, pred_test_y = ml.map_loss_in_batches(
            map_and_loss_return_map,
            trained_model,
            test_X,
            test_Y,
            args.batch,
            subkey1,
            devices=jax.devices(),
            return_map=True,
        )
        results[t, k, 2] = test_loss
        print(f"{t},{model_name}: {results[t,k,2]}")

print("Test Errors")
for k, (model_name, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
