import sys
import time
import numpy as np
from typing_extensions import Optional, Self, Union
import wandb
import argparse
import functools as ft

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils

import signax


def get_dataset(
    D: int,
    n_curves: int,
    integrator_steps: int,
    sig_order: int,
    subsample_steps: int,
    key: ArrayLike,
) -> tuple[jax.Array, list[jax.Array]]:
    t = jnp.linspace(0, 1, num=integrator_steps)
    library = jnp.stack([jnp.ones_like(t), t, t**2, t**3, t**4, t**5], axis=-1)  # (steps, library)
    coeffs = random.normal(key, shape=(n_curves, library.shape[1], D))  # (batch,library,channels)
    curves = jnp.einsum("ij,bjc->bic", library, coeffs)  # (batch,steps,channels)

    signature = signax.signature(curves, sig_order)
    return curves[:, :: (integrator_steps // subsample_steps)], signature


def get_data(
    D: int, n_train: int, n_val: int, n_test: int, sig_order: int, steps: int, key: ArrayLike
) -> jax.Array:
    integrator_steps = 1000
    subkey1, subkey2, subkey3 = random.split(key, num=3)

    train_X, train_Y = get_dataset(D, n_train, integrator_steps, sig_order, steps, subkey1)
    val_X, val_Y = get_dataset(D, n_val, integrator_steps, sig_order, steps, subkey2)
    test_X, test_Y = get_dataset(D, n_test, integrator_steps, sig_order, steps, subkey3)

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

    net: eqx.Module

    max_k: int

    def __init__(
        self: Self, D: int, max_k: int, steps: int, width: int, depth: int, key: ArrayLike
    ) -> Self:
        self.max_k = max_k
        assert max_k <= 3, "EquivSignature currently can only handle up to max_k=3"

        n_in = steps * D
        n_out = sum([D**k for k in range(1, 1 + max_k)])

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.net(x.reshape(-1))


class EquivSignature(eqx.Module):

    net: eqx.Module

    max_k: int

    def __init__(
        self: Self, max_k: int, steps: int, width: int, depth: int, key: ArrayLike
    ) -> Self:
        self.max_k = max_k
        assert max_k <= 3, "EquivSignature currently can only handle up to max_k=3"

        n_in = steps + steps * (steps - 1) // 2
        n_out = sum([steps**k for k in range(1, 1 + max_k)])
        n_out = n_out + 1 if max_k >= 2 else n_out

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        n, D = x.shape

        id = jnp.eye(D)
        id = id.at[0, 0].set(-1)

        X = (x @ id @ x.T)[jnp.triu_indices(n)].reshape(-1)
        assert X.shape == (n + n * (n - 1) // 2,)

        all_coeffs = self.net(X)

        out = jnp.zeros(0)
        idx = 0
        for k in range(1, 1 + self.max_k):
            ein_str = ",".join([f"{utils.LETTERS[i] + utils.LETTERS[i+13]}" for i in range(k)])
            ein_str += f"->{utils.LETTERS[:k] + utils.LETTERS[13:13+k]}"
            tensor_inputs = (x,) * k
            basis = jnp.einsum(ein_str, *tensor_inputs).reshape((n**k,) + (D,) * k)
            if k == 2:
                basis = jnp.concatenate([basis, jnp.eye(D)[None]])

            collapsed_basis = basis.reshape((len(basis), D**k))
            coeffs = all_coeffs[idx : idx + len(basis)]
            out = jnp.concatenate([out, coeffs @ collapsed_basis])
            idx += len(basis)

        assert idx == len(all_coeffs)

        return out


def map_and_loss(
    model: eqx.Module, x: jax.Array, y: jax.Array, aux_data: Optional[eqx.nn.State]
) -> jax.Array:
    """

    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
    """
    batch, _, D = x.shape
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
        "--wandb",
        help="whether to use wandb on this run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-entity", help="the wandb user", type=str, default="wilson_gregory")
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="signature-tensors-lorentz-2"
    )

    return parser.parse_args()


# MAIN
args = handleArgs(sys.argv)
D = 4
sig_order = 3
steps = 10

key = random.PRNGKey(time.time_ns() if args.seed is None else args.seed)
key, subkey = random.split(key)
train_X, train_Y, val_X, val_Y, test_X, test_Y = get_data(
    D, args.n_train, args.n_val, args.n_test, sig_order, steps, key
)

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    ("tensor_poly", 5e-4, True, EquivSignature(sig_order, steps, 32, 3, subkey2)),
    ("baseline_samewidth", 5e-3, True, Baseline(D, sig_order, steps, 32, 3, subkey4)),
    ("baseline_sameparams", 1e-3, True, Baseline(D, sig_order, steps, 128, 3, subkey3)),
    ("signax_only", 1, True, SignaxOnly(sig_order)),
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
                            args.epochs * args.n_train // args.batch, lr, div_factor=3
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
        results[t, k, 2] = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            test_X,
            test_Y,
            args.batch,
            subkey1,
            devices=jax.devices(),
        )
        print(f"{t},{model_name}: {results[t,k,2]}")

print("Test Errors")
for k, (model_name, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
