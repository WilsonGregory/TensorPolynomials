import functools as ft
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
    data_type: str,
    data_degree: int,
    D: int,
    n_train: int,
    n_val: int,
    n_test: int,
    sig_order: int,
    top_sig_only: bool,
    steps: int,
    key: ArrayLike,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, list[int]]:
    subkey1, subkey2, subkey3 = random.split(key, num=3)

    # train_X, train_Y = data.get_signature_old(D, n_train, 1000, sig_order, steps, subkey1)
    # val_X, val_Y = data.get_signature_old(D, n_val, 1000, sig_order, steps, subkey2)
    # test_X, test_Y = data.get_signature_old(D, n_test, 1000, sig_order, steps, subkey3)

    library_type = data_type  # for now, may have an option to use the old ones as well
    train_X, train_Y = data.get_signature_flat(
        D,
        n_train,
        sig_order,
        top_sig_only,
        steps,
        (0, 1),
        (library_type, data_degree),
        subkey1,
        coeffs_dist="uniform",
    )
    val_X, val_Y = data.get_signature_flat(
        D,
        n_val,
        sig_order,
        top_sig_only,
        steps,
        (0, 1),
        (library_type, data_degree),
        subkey2,
        coeffs_dist="uniform",
    )
    test_X, test_Y = data.get_signature_flat(
        D,
        n_test,
        sig_order,
        top_sig_only,
        steps,
        (0, 1),
        (library_type, data_degree),
        subkey3,
        coeffs_dist="uniform",
    )

    orders = [sig_order] if top_sig_only else list(range(1, sig_order + 1))

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, orders


def count_params(model) -> int:
    return sum(
        [
            0 if x is None else x.size
            for x in eqx.filter(jax.tree_util.tree_leaves(model), eqx.is_array)
        ]
    )


class SignaxOnly(eqx.Module):
    D: int
    max_k: int
    sig_orders: list[int]

    def __init__(self: Self, D: int, sig_orders: list[int]):
        self.D = D
        self.sig_orders = sig_orders
        self.max_k = max(*sig_orders) if len(sig_orders) > 1 else sig_orders[0]

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        # add batch dim, calc signature, them remove batch dim
        signature_full_flat = signax.signature(x[None], self.max_k)
        full_orders = list(range(1, self.max_k + 1))
        if self.sig_orders != full_orders:
            full_signature = data.expand_signature(self.D, signature_full_flat, full_orders)
            signature = {k: full_signature[k] for k in self.sig_orders}
            signature_flat = data.flatten_signature(signature)
        else:
            signature_flat = signature_full_flat

        return signature_flat[0]


class Baseline(eqx.Module):

    net: eqx.nn.MLP

    sig_orders: list[int]

    def __init__(
        self: Self,
        D: int,
        sig_orders: list[int],
        steps: int,
        width: int,
        depth: int,
        key: jax.Array,
    ) -> None:
        self.sig_orders = sig_orders
        n_in = steps * D
        n_out = sum([D**k for k in sig_orders])

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.net(x.reshape(-1))


class EquivSignature(eqx.Module):

    net: eqx.nn.MLP

    sig_orders: list[int]
    max_k: int
    bilipschitz: bool
    eps: float

    def __init__(
        self: Self,
        sig_orders: list[int],
        steps: int,
        width: int,
        depth: int,
        bilipschitz: bool,
        key: jax.Array,
        eps: float = 1e-4,
    ) -> None:
        self.sig_orders = sig_orders
        self.max_k = max(*sig_orders) if len(sig_orders) > 1 else sig_orders[0]
        self.bilipschitz = bilipschitz
        self.eps = eps

        n_in = steps + steps * (steps - 1) // 2
        n_out = sum([steps**k for k in sig_orders])
        n_out += utils.metric_tensor_basis_size(sig_orders, steps)

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
        kth_basis = utils.get_tensor_basis_of_vecs(x, self.max_k, jnp.eye(D))

        for k in self.sig_orders:
            # scale basis elements so they have close to unit norm
            collapsed_basis = kth_basis[k].reshape((-1, D**k)) / ((D ** (k / 2)) * (k**2))
            coeffs = all_coeffs[idx : idx + len(collapsed_basis)]
            out = jnp.concatenate([out, coeffs @ collapsed_basis])
            idx += len(collapsed_basis)

        assert idx == len(all_coeffs)

        return out


class Normalizer:

    INPUT_TYPES = ["norm", "gram", "standard"]
    OUTPUT_TYPES = ["norm", "Chevyrev-Oberhauser", "align_basis", "mean_norm", "standard"]

    D: int
    input_type: str | None
    output_type: str | None
    n_train: int
    input_scale: jax.Array | float
    output_scale: dict[int, jax.Array]
    orders: list[int]

    def __init__(
        self: Self,
        D: int,
        input_type: str | None,
        output_type: str | None,
        n_train: int,
        orders: list[int],
    ) -> None:
        self.D = D
        self.input_type = input_type
        self.output_type = output_type
        self.n_train = n_train
        self.orders = orders

    def input(self: Self, vectors: jax.Array, *extra_vectors: jax.Array) -> tuple[jax.Array, ...]:
        """
        Normalize the input vectors.

        args:
            vectors: input train vectors, shape (batch,steps,D)
            extra_vectors: input vectors for the test or validation set, normalize based on train set
        """
        if self.input_type == "norm":
            self.input_scale = jnp.std(jnp.linalg.norm(vectors), axis=-1)
        elif self.input_type == "gram":
            batch, steps, _ = vectors.shape

            # (b,steps,steps)
            X = jnp.einsum("...ij,...kj->...ik", vectors, vectors)
            X = X[:, jnp.triu_indices(steps)].reshape((batch, -1))  # (b,gram matrix dim)
            # sqrt of std because we are applying it to each vector
            self.input_scale = jnp.sqrt(jnp.std(X))
        elif self.input_type == "standard":
            # this won't get inverted, but if we are doing loss differences it shouldn't matter
            vec_mean = jnp.mean(vectors)
            vectors -= vec_mean
            extra_vectors = tuple(vecs - vec_mean for vecs in extra_vectors)
            self.input_scale = jnp.std(vectors)
        elif self.input_type is None:
            self.input_scale = 1
        else:
            raise ValueError

        return vectors / self.input_scale, *[vecs / self.input_scale for vecs in extra_vectors]

    def inverse_input(self: Self, vectors: jax.Array) -> jax.Array:
        return vectors * self.input_scale

    def output(
        self: Self,
        signature_flat: jax.Array,
        vectors: jax.Array | None,
        *extra_signatures_flat: jax.Array,
    ) -> tuple[jax.Array, ...]:
        """
        Normalize the path signature in list form.

        args:
            flat_signature: list of jax arrays of shape (batch, (D,)*k)
        """
        extra_signatures = [
            data.expand_signature(self.D, sig_flat, self.orders)
            for sig_flat in extra_signatures_flat
        ]
        signature = data.expand_signature(self.D, signature_flat, self.orders)
        if self.output_type == "standard":
            # this won't get inverted, but if we are doing loss differences it shouldn't matter
            sig_mean = jnp.mean(signature_flat)
            signature_flat -= sig_mean
            extra_signatures_flat = tuple(
                extra_sig - sig_mean for extra_sig in extra_signatures_flat
            )
            self.output_scale = {k: jnp.std(signature_flat) for k in self.orders}

        if self.output_type == "align_basis":
            assert vectors is not None
            # scale the output tensors so that the standard deviations of their norm match
            # the standard deviation of the norm of the basis elements
            vmap_tensor_basis = jax.vmap(utils.get_tensor_basis_of_vecs, in_axes=(0, None, None))
            kth_basis = vmap_tensor_basis(vectors, len(signature), jnp.eye(D))

            self.output_scale = {}
            for k, t in signature.items():
                std_basis = jnp.std(jnp.linalg.norm(kth_basis[k].reshape((-1, D**k)), axis=-1))
                std_t = jnp.std(jnp.linalg.norm(t.reshape((self.n_train, -1)), axis=-1))
                self.output_scale[k] = std_t / std_basis

        elif self.output_type == "mean_norm":
            # scale the tensor signatures so that their expected norm is 1.
            self.output_scale = {}
            for k, t in signature.items():
                self.output_scale[k] = jnp.mean(
                    jnp.linalg.norm(t.reshape((self.n_train, -1)), axis=-1)
                )

        elif self.output_type is None:
            self.output_scale = {k: jnp.ones(1) for k in self.orders}
        elif self.output_type == "standard":
            pass  # implemented above
        else:
            raise ValueError

        scaled_signature = {k: t / self.output_scale[k] for k, t in signature.items()}
        scaled_signature_flat = data.flatten_signature(scaled_signature)

        # scale the extra signatures, then flatten them
        scaled_extra_sigs_flat = []
        for extra_signature in extra_signatures:
            scaled_extra_signature = {
                k: t / self.output_scale[k] for k, t in extra_signature.items()
            }
            scaled_extra_sigs_flat.append(data.flatten_signature(scaled_extra_signature))

        return scaled_signature_flat, *scaled_extra_sigs_flat

    def inverse_output(self: Self, signature_flat: jax.Array) -> jax.Array:
        signature = data.expand_signature(self.D, signature_flat, self.orders)
        scaled_signature = {k: t * self.output_scale[k] for k, t in signature.items()}
        return data.flatten_signature(scaled_signature)


def map_and_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
    sig_orders: list[int],
    normalizer: Normalizer | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    Map and loss that optionally unnormalizes the data to compare between models with different
    normalization schemes.

    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
        aux_data: auxilliary data used in the model layers like batch norm
        normalizer: object to unnormalize the output so all losses are on the same scale
    """
    batch, _, D = x.shape
    assert callable(model)
    pred_signature_flat = jax.vmap(model)(x)

    if normalizer is not None:
        pred_signature_flat = normalizer.inverse_output(pred_signature_flat)
        y = normalizer.inverse_output(y)

    pred_signature = data.expand_signature(D, pred_signature_flat, sig_orders)
    y_signature = data.expand_signature(D, y, sig_orders)

    loss_per_tensor = jnp.zeros((batch, 0))
    for pred_tensor, y_tensor in zip(pred_signature.values(), y_signature.values()):
        pred_tensor, y_tensor = pred_tensor.reshape((batch, -1)), y_tensor.reshape((batch, -1))
        loss_per_tensor = jnp.concatenate(
            [
                loss_per_tensor,
                jnp.mean((pred_tensor - y_tensor) ** 2, axis=1, keepdims=True),
            ],
            axis=1,
        )

    # mean over batch and tensor types. And previously we had mean over tensor entries, so there
    # error in the 1-tensor is balanced with the error in the 2-tensor.
    return jnp.mean(loss_per_tensor), aux_data


def map_and_loss_return_map(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
    sig_orders: list[int],
    normalizer: Normalizer | None = None,
) -> tuple[jax.Array, eqx.nn.State | None, jax.Array]:
    """
    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
    """
    batch, _, D = x.shape
    assert callable(model)
    pred_signature_flat = jax.vmap(model)(x)

    if normalizer is not None:
        pred_signature_flat = normalizer.inverse_output(pred_signature_flat)
        y = normalizer.inverse_output(y)

    pred_signature = data.expand_signature(D, pred_signature_flat, sig_orders)
    y_signature = data.expand_signature(D, y, sig_orders)

    loss_per_tensor = jnp.zeros((batch, 0))
    for pred_tensor, y_tensor in zip(pred_signature.values(), y_signature.values()):
        pred_tensor, y_tensor = pred_tensor.reshape((batch, -1)), y_tensor.reshape((batch, -1))
        loss_per_tensor = jnp.concatenate(
            [
                loss_per_tensor,
                jnp.mean((pred_tensor - y_tensor) ** 2, axis=1, keepdims=True),
            ],
            axis=1,
        )

    # mean over batch and tensor types. And previously we had mean over tensor entries, so there
    # error in the 1-tensor is balanced with the error in the 2-tensor.
    return jnp.mean(loss_per_tensor), aux_data, pred_signature_flat


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        help="data to use",
        choices=["polynomial", "piecewise_linear"],
        type=str,
        default="polynomial",
    )
    parser.add_argument(
        "--data-degree",
        help="degree of polynomial or piecewise linear function to use",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--top-sig-only",
        help="whether to only use the highest order tensor",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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
train_X, train_sig, val_X, val_sig, test_X, test_sig, sig_orders = get_data(
    args.data,
    args.data_degree,
    D,
    args.n_train,
    args.n_val,
    args.n_test,
    args.sig_order,
    args.top_sig_only,
    steps,
    key,
)

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    (
        "signax_only",
        1,
        SignaxOnly(D, sig_orders),
        Normalizer(D, None, None, args.n_train, sig_orders),
    ),
    (
        "tensor_poly",
        1e-3,  # 1e-3 for piecewise_linear
        EquivSignature(sig_orders, steps, 32, 3, False, subkey1),
        Normalizer(D, "gram", "mean_norm", args.n_train, sig_orders),
    ),
    (
        "baseline_samewidth",
        5e-3,
        Baseline(D, sig_orders, steps, 32, 3, subkey4),
        Normalizer(D, "standard", "standard", args.n_train, sig_orders),
    ),
    (
        "baseline_sameparams",
        1e-3,
        Baseline(D, sig_orders, steps, 128, 3, subkey3),
        Normalizer(D, "standard", "standard", args.n_train, sig_orders),
    ),
]

results = np.zeros((args.n_trials, len(models_list), 3))
for t in range(args.n_trials):
    for k, (model_name, lr, model, normalizer) in enumerate(models_list):

        print(f"{model_name}: {count_params(model):,} params")

        name = f"signature_{model_name}_t{t}_lr{lr}_e{args.epochs}_ntrain{args.n_train}"
        save_model = f"{args.save_model}/{name}.eqx" if args.save_model is not None else None

        train_X_norm, val_X_norm, test_X_norm = normalizer.input(train_X, val_X, test_X)
        train_sig_norm, val_sig_norm, test_sig_norm = normalizer.output(
            train_sig, train_X_norm, val_sig, test_sig
        )

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
                        "normalize_input": normalizer.input_type,
                        "normalize_output": normalizer.output_type,
                    }
                )

            steps_per_epoch = int(args.n_train / args.batch)

            # normalizer values will be set for val set, not train
            key, subkey = random.split(key)
            trained_model, _, _, _ = ml.train(
                train_X_norm,
                train_sig_norm,
                ft.partial(map_and_loss, sig_orders=sig_orders),
                model,
                subkey,
                ml.EpochStop(args.epochs, verbose=args.verbose),
                # ml.ValLoss(patience=100, max_epochs=args.epochs, verbose=args.verbose),
                args.batch,
                optax.adamw(
                    optax.cosine_onecycle_schedule(args.epochs * steps_per_epoch, lr, div_factor=3)
                ),
                validation_X=val_X_norm,
                validation_Y=val_sig_norm,
                val_map_and_loss=ft.partial(
                    map_and_loss, sig_orders=sig_orders, normalizer=normalizer
                ),
                save_model=save_model,
                is_wandb=args.wandb,
            )

            if args.wandb:
                wandb.finish()

            if args.save_model is not None:
                ml.save(save_model, trained_model)

        key, subkey1, subkey2, subkey3 = random.split(key, num=4)
        results[t, k, 0] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            train_X_norm,
            train_sig_norm,
            args.batch,
            subkey1,
            devices=jax.devices(),
        )
        results[t, k, 1] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            val_X_norm,
            val_sig_norm,
            args.batch,
            subkey2,
            devices=jax.devices(),
        )
        results[t, k, 2] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            test_X_norm,
            test_sig_norm,
            args.batch,
            subkey3,
            devices=jax.devices(),
        )
        print(f"{t},{model_name}: {results[t,k,2]}")

print("Test Errors")
for k, (model_name, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
