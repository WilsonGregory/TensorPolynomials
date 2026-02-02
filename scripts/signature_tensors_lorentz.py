import sys
import time
import functools as ft
import numpy as np
from typing_extensions import Self
import wandb
import argparse

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import optax
import equinox as eqx

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils
import tensorpolynomials.data as data

import signax


def get_data(
    D: int, n_train: int, n_val: int, n_test: int, sig_order: int, steps: int, key: ArrayLike
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    integrator_steps = 1000
    subkey1, subkey2, subkey3 = random.split(key, num=3)

    train_X, train_Y = data.get_signature_old(
        D, n_train, integrator_steps, sig_order, steps, subkey1
    )
    val_X, val_Y = data.get_signature_old(D, n_val, integrator_steps, sig_order, steps, subkey2)
    test_X, test_Y = data.get_signature_old(D, n_test, integrator_steps, sig_order, steps, subkey3)

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
        assert max_k <= 3, "EquivSignature currently can only handle up to max_k=3"

        n_in = steps * D
        n_out = sum([D**k for k in range(1, 1 + max_k)])

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.net(x.reshape(-1))


class EquivSignature(eqx.Module):

    net: eqx.nn.MLP
    sig_orders: list[int]
    max_k: int

    def __init__(
        self: Self, sig_orders: list[int], steps: int, width: int, depth: int, key: jax.Array
    ) -> None:
        self.sig_orders = sig_orders
        self.max_k = max(*sig_orders) if len(sig_orders) > 1 else sig_orders[0]

        n_in = steps + steps * (steps - 1) // 2
        n_out = sum([steps**k for k in sig_orders])
        n_out += utils.metric_tensor_basis_size(sig_orders, steps)

        self.net = eqx.nn.MLP(n_in, n_out, width, depth, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        n, D = x.shape

        # the metric tensor for the Lorentz group
        id = jnp.eye(D)
        id = id.at[0, 0].set(-1)

        X = (x @ id @ x.T)[jnp.triu_indices(n)].reshape(-1)
        assert X.shape == (n + n * (n - 1) // 2,)

        all_coeffs = self.net(X)

        out = jnp.zeros(0)
        idx = 0
        kth_basis = utils.get_tensor_basis_of_vecs(x, self.max_k, id)

        for k in self.sig_orders:
            # scale basis elements so they have close to unit norm
            collapsed_basis = kth_basis[k].reshape((-1, D**k)) / ((D ** (k / 2)) * (k**2))
            coeffs = all_coeffs[idx : idx + len(collapsed_basis)]
            out = jnp.concatenate([out, coeffs @ collapsed_basis])
            idx += len(collapsed_basis)

        assert idx == len(all_coeffs)

        return out


# TODO: for lorentz this needs to be updated
def augment_data(
    train_X: jax.Array,
    train_sig: jax.Array,
    aug_multiplier: int,
    orders: list[int],
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch, steps, D = train_X.shape

    train_X_aug = jnp.full((aug_multiplier,) + train_X.shape, train_X[None])
    train_X_aug = train_X_aug.reshape((batch * aug_multiplier, steps, D))

    train_sig_aug = jnp.full((aug_multiplier,) + train_sig.shape, train_sig[None])
    train_sig_aug = train_sig_aug.reshape((batch * aug_multiplier,) + train_sig.shape[1:])

    key, subkey = random.split(key)
    Qs = random.orthogonal(subkey, D, (batch * aug_multiplier,))
    train_X_aug_rot = jnp.einsum(
        "...ij,...j->...i",
        jnp.full((len(Qs), steps, D, D), Qs[:, None]),
        train_X_aug,
        precision=jax.lax.Precision.HIGHEST,
    )

    train_sig_aug_rot = {}
    for k, tensor in data.expand_signature(D, train_sig_aug, orders).items():
        assert k > 0

        einstr = "".join([f"...{utils.LETTERS[i]}{utils.LETTERS[i+26]}," for i in range(k)])
        einstr += f"...{utils.LETTERS[26:26+k]}->...{utils.LETTERS[:k]}"
        tensor_rot = jnp.einsum(einstr, *((Qs,) * k), tensor, precision=jax.lax.Precision.HIGHEST)
        train_sig_aug_rot[k] = tensor_rot

    return train_X_aug_rot, data.flatten_signature(train_sig_aug_rot)


class Normalizer:

    INPUT_TYPES = ["norm", "gram", "standard"]
    OUTPUT_TYPES = ["norm", "Chevyrev-Oberhauser", "align_basis", "mean_norm", "standard"]

    D: int
    input_type: str | None
    output_type: str | None
    input_scale: jax.Array | float
    output_scale: dict[int, jax.Array]
    orders: list[int]

    def __init__(
        self: Self,
        D: int,
        input_type: str | None,
        output_type: str | None,
        orders: list[int],
    ) -> None:
        self.D = D
        self.input_type = input_type
        self.output_type = output_type
        self.orders = orders

    def input(self: Self, vectors: jax.Array, *extra_vectors: jax.Array) -> tuple[jax.Array, ...]:
        """
        Normalize the input vectors.

        args:
            vectors: input train vectors, shape (batch,steps,D)
            extra_vectors: input vectors for the test or validation set, normalize based on train set
        """
        if self.input_type == "norm":
            self.input_scale = jnp.std(jnp.linalg.norm(vectors, axis=-1))
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
        batch = len(signature_flat)
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

        if self.output_type == "norm":
            # scale the tensor signatures so that their expected norm is 1.
            self.output_scale = {}
            for k, t in signature.items():
                self.output_scale[k] = jnp.std(jnp.linalg.norm(t.reshape((batch, -1)), axis=-1))
        elif self.output_type == "align_basis":
            assert vectors is not None
            # scale the output tensors so that the standard deviations of their norm match
            # the standard deviation of the norm of the basis elements
            vmap_tensor_basis = jax.vmap(utils.get_tensor_basis_of_vecs, in_axes=(0, None, None))
            kth_basis = vmap_tensor_basis(vectors, len(signature), jnp.eye(D))

            self.output_scale = {}
            for k, t in signature.items():
                std_basis = jnp.std(jnp.linalg.norm(kth_basis[k].reshape((-1, D**k)), axis=-1))
                std_t = jnp.std(jnp.linalg.norm(t.reshape((batch, -1)), axis=-1))
                self.output_scale[k] = std_t / std_basis

        elif self.output_type == "mean_norm":
            # scale the tensor signatures so that their expected norm is 1.
            self.output_scale = {}
            for k, t in signature.items():
                self.output_scale[k] = jnp.mean(jnp.linalg.norm(t.reshape((batch, -1)), axis=-1))

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
sig_orders = list(range(1, 1 + sig_order))
steps = 10

key = random.PRNGKey(time.time_ns() if args.seed is None else args.seed)
key, subkey = random.split(key)
train_X, train_sig, val_X, val_sig, test_X, test_sig = get_data(
    D, args.n_train, args.n_val, args.n_test, sig_order, steps, key
)

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    # previous model `tensor_poly` didn't include full basis
    (
        "tensor_poly32",
        5e-4,
        EquivSignature(sig_orders, steps, 32, 3, subkey1),
        1,
        Normalizer(D, "gram", "mean_norm", sig_orders),
    ),
    (
        "baseline_samewidth",
        5e-3,
        Baseline(D, sig_order, steps, 32, 3, subkey2),
        1,
        Normalizer(D, "standard", "standard", sig_orders),
    ),
    (
        "baseline_sameparams116",
        1e-3,
        Baseline(D, sig_order, steps, 116, 3, subkey3),
        1,
        Normalizer(D, "standard", "standard", sig_orders),
    ),
    (
        "baseline_aug",
        1e-3,
        Baseline(D, sig_order, steps, 116, 3, subkey4),
        4,
        Normalizer(D, "norm", "norm", sig_orders),
    ),
    ("signax_only", 1, SignaxOnly(sig_order), 1, Normalizer(D, None, None, sig_orders)),
]

results = np.zeros((args.n_trials, len(models_list), 3))
for t in range(args.n_trials):
    for k, (model_name, lr, model, aug_multiplier, normalizer) in enumerate(models_list):
        key, subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, num=6)

        print(f"{model_name}: {count_params(model):,} params")

        name = f"signature_{model_name}_t{t}_lr{lr}_e{args.epochs}_ntrain{args.n_train}"
        save_model = f"{args.save_model}/{name}.eqx" if args.save_model is not None else None

        if aug_multiplier > 1:
            train_X_aug, train_sig_aug = augment_data(
                train_X, train_sig, aug_multiplier, sig_orders, subkey1
            )
        else:
            train_X_aug, train_sig_aug = train_X, train_sig

        train_X_norm, val_X_norm, test_X_norm = normalizer.input(train_X_aug, val_X, test_X)
        train_sig_norm, val_sig_norm, test_sig_norm = normalizer.output(
            train_sig_aug, train_X_norm, val_sig, test_sig
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

            steps_per_epoch = int(len(train_X_norm) / args.batch)

            # normalizer values will be set for val set, not train
            trained_model, _, _, _ = ml.train(
                train_X_norm,
                train_sig_norm,
                ft.partial(map_and_loss, sig_orders=sig_orders),
                model,
                subkey2,
                ml.EpochStop(args.epochs, verbose=args.verbose),
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

        results[t, k, 0] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            train_X_norm,
            train_sig_norm,
            args.batch,
            subkey3,
            devices=jax.devices(),
        )
        results[t, k, 1] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            val_X_norm,
            val_sig_norm,
            args.batch,
            subkey4,
            devices=jax.devices(),
        )
        results[t, k, 2] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, sig_orders=sig_orders, normalizer=normalizer),
            trained_model,
            test_X_norm,
            test_sig_norm,
            args.batch,
            subkey5,
            devices=jax.devices(),
        )
        print(f"{t},{model_name}: {results[t,k,2]}")

print("Test Errors")
for k, (model_name, _, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
