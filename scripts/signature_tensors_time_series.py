import functools as ft
import math
import sys
import time
import numpy as np
from typing_extensions import Self
import wandb
import argparse
import scipy.optimize

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import equinox as eqx

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils

import signax

from aeon.datasets import load_classification

UWAVE = "UWaveGestureLibrary"

SYMMETRY_BREAKERS_FULL = "full"


def psi_norm(sig_norms: jax.Array, C: int, a: int):
    """
    psi function derived using corollary 15 https://arxiv.org/pdf/1810.10971

    args:
        sig_norms: norms of the signatures, shape (b,)
        C: constant, paper says 4
        a: constant, paper says 1
    """
    return jnp.where(
        sig_norms <= C * jnp.ones_like(sig_norms),
        sig_norms,
        C + (C ** (1 + a)) * (C ** (-a) + sig_norms ** (-a)) / a,
    )


def lambda_norm(signature: list[jax.Array], C: float, a: int):
    """
    lambda map from derived using corollary 15 https://arxiv.org/pdf/1810.10971

    args:
        signature: flattened signature, shape (batch,flattened sig). does not include scalar
        C: constant, paper says 4
        a: constant, paper says 1
    """
    batch = len(signature[0])
    signature = [jnp.ones((batch, 1))] + signature  # add the scalar to the signature

    sig_tensor_norms = jnp.stack(
        [jnp.linalg.norm(t.reshape((batch, -1)), axis=-1) for t in signature], axis=-1
    )  # (b,sig_order)

    sig_norms_squared = jnp.sum(sig_tensor_norms**2, axis=-1)  # (b,)
    if C == -1:
        # for UWaveGestureLibrary roughly 14% of the data is above this
        C = int(jnp.mean(sig_norms_squared) + jnp.std(sig_norms_squared))
    elif 0 < C < 1:  # C is a percentile of the data
        C = int(jnp.percentile(sig_norms_squared, C * 100))

    print(f"Signature squared norm: {jnp.mean(sig_norms_squared < C)*100:.2f}% is <{C}")
    psi_norms = psi_norm(sig_norms_squared, int(C), a)  # need to square this so that the root is 1

    roots = []
    for row_sig_tensor_norms, row_psi in zip(sig_tensor_norms, psi_norms):
        row_norms = [float(x) for x in row_sig_tensor_norms]  # convert to list
        f = (
            lambda x: sum([(x ** (2 * m)) * t_norm**2 for m, t_norm in enumerate(row_norms)])
            - row_psi
        )
        roots.append(scipy.optimize.brentq(f, 0, 2))

    roots = jnp.stack(roots)  # (b,)
    roots = jnp.where(roots > 1, jnp.ones_like(roots), roots)  # can only shrink, not grow
    normalized_sig = [
        t * (roots.reshape((batch,) + (1,) * m) ** m) for m, t in enumerate(signature)
    ]

    return normalized_sig[1:]  # drop the initial scalar, which is always 1


def get_data(
    sig_order: int,
    subsample_steps: int,
    n_train: int,
    n_val: int,
    n_test: int,
    symmetry_breakers: str | None,
    one_hot_labels: bool = True,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    D = 3
    # data shape (batch,1,steps), labels shape (batch,)
    x, labels, meta_data = load_classification("UWaveGestureLibraryX", return_metadata=True)
    labels = jnp.array(labels.astype(int))  # (b,)
    y, _ = load_classification("UWaveGestureLibraryY")
    z, _ = load_classification("UWaveGestureLibraryZ")
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, np.ndarray)

    curves = jnp.moveaxis(jnp.concatenate([x, y, z], axis=1), 1, 2)  # (b,D,steps) -> (b,steps,D)
    curves -= curves[:, :1]  # path signature moves each path to begin at 0, so maybe we should too?

    batch, steps, _ = curves.shape
    assert n_train + n_val + n_test < len(curves)

    signature_flat = signax.signature(curves, sig_order, flatten=True)

    # (b,sub_steps,D)
    sample_points = curves[:, :: (steps // subsample_steps)][:, :subsample_steps]

    if symmetry_breakers == SYMMETRY_BREAKERS_FULL:
        mean_length = jnp.mean(jnp.linalg.norm(sample_points, axis=-1))  # scale them to match
        directional_vectors = jnp.full((batch, D, D), jnp.eye(D)[None] * mean_length)
        sample_points = jnp.concatenate([sample_points, directional_vectors], axis=1)

    if one_hot_labels:
        # convert to one hot encoded labels, shape (b,8)
        labels = jnp.stack([(labels == i).astype(float) for i in jnp.unique(labels)], axis=-1)

    start = 0
    stop = n_train
    train_X = sample_points[start:stop]
    train_Y = signature_flat[start:stop]
    train_labels = labels[start:stop]

    start = n_train
    stop = n_train + n_val
    val_X = sample_points[start:stop]
    val_Y = signature_flat[start:stop]
    val_labels = labels[start:stop]

    start = n_train + n_val
    stop = n_train + n_val + n_test
    test_X = sample_points[start:stop]
    test_Y = signature_flat[start:stop]
    test_labels = labels[start:stop]

    return train_X, train_Y, train_labels, val_X, val_Y, val_labels, test_X, test_Y, test_labels


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

        # default parameter initialization. I tested kaiming initialization which helps train
        # loss, but hurts the validation loss (overfitting).
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
        kth_basis = utils.get_tensor_basis_of_vecs(x, self.max_k, jnp.eye(D))

        for k in range(1, 1 + self.max_k):
            # scale basis elements so they have close to unit norm
            collapsed_basis = kth_basis[k].reshape((-1, D**k)) / ((D ** (k / 2)) * (k**2))
            coeffs = all_coeffs[idx : idx + len(collapsed_basis)]
            out = jnp.concatenate([out, coeffs @ collapsed_basis])
            idx += len(collapsed_basis)

        assert idx == len(all_coeffs)

        return out


def expand_signature(D: int, signature_flat: jax.Array) -> list[jax.Array]:
    """
    Given a flat signature, expand it into list form. Signature starts at k=1

    args:
        signature_flat: shape (batch,flat_signature)
    """
    idx = 0
    k = 1
    signature = []
    while idx < signature_flat.shape[1]:
        signature.append(signature_flat[:, idx : idx + D**k].reshape((-1,) + (D,) * k))
        idx += D**k
        k += 1

    assert idx == signature_flat.shape[1]

    return signature


class Normalizer:

    INPUT_TYPES = ["norm", "gram", "standard"]
    OUTPUT_TYPES = ["norm", "Chevyrev-Oberhauser", "align_basis", "standard"]

    D: int
    input_type: str | None
    output_type: str | None
    n_train: int
    input_scale: jax.Array | float
    output_scale: jax.Array

    # constants used for Chevyrev-Oberhauser
    C: int
    a: int

    def __init__(
        self: Self,
        D: int,
        input_type: str | None,
        output_type: str | None,
        n_train: int,
        C: int,
        a: int,
    ) -> None:
        self.D = D
        self.input_type = input_type
        self.output_type = output_type
        self.n_train = n_train
        self.C = C
        self.a = a

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
            expand_signature(self.D, sig_flat) for sig_flat in extra_signatures_flat
        ]
        if self.output_type == "standard":
            # this won't get inverted, but if we are doing loss differences it shouldn't matter
            sig_mean = jnp.mean(signature_flat)
            signature_flat -= sig_mean
            extra_signatures_flat = tuple(
                extra_sig - sig_mean for extra_sig in extra_signatures_flat
            )

            signature = expand_signature(self.D, signature_flat)
            self.output_scale = jnp.ones(len(signature)) * jnp.std(signature_flat)
        else:
            signature = expand_signature(self.D, signature_flat)

        if self.output_type == "Chevyrev-Oberhauser":
            # currently cant inverse this one, so just return here
            scaled_signature = lambda_norm(signature, self.C, self.a)
            scaled_signature_flat = jnp.concatenate(
                [t.reshape((len(t), -1)) for t in scaled_signature], axis=-1
            )

            scaled_extra_sigs_flat = []
            for extra_signature in extra_signatures:
                scaled_extra_signature = lambda_norm(extra_signature, self.C, self.a)
                scaled_extra_sigs_flat.append(
                    jnp.concatenate(
                        [t.reshape((len(t), -1)) for t in scaled_extra_signature], axis=-1
                    )
                )
            return scaled_signature_flat, *scaled_extra_sigs_flat

        if self.output_type == "norm":
            self.output_scale = jnp.stack(
                [
                    jnp.std(jnp.linalg.norm(t.reshape((self.n_train, -1)), axis=-1))
                    for t in signature
                ]
            )
        elif self.output_type == "align_basis":
            assert vectors is not None
            # scale the output tensors so that the standard deviations of their norm match
            # the standard deviation of the norm of the basis elements
            vmap_tensor_basis = jax.vmap(utils.get_tensor_basis_of_vecs, in_axes=(0, None, None))
            kth_basis = vmap_tensor_basis(vectors, len(signature), jnp.eye(D))

            output_scale = []
            for k, t, basis_t in zip(range(1, len(signature) + 1), signature, kth_basis.values()):
                std_basis = jnp.std(jnp.linalg.norm(basis_t.reshape((-1, D**k)), axis=-1))
                std_t = jnp.std(jnp.linalg.norm(t.reshape((self.n_train, -1)), axis=-1))
                output_scale.append(std_t / std_basis)

            self.output_scale = jnp.stack(output_scale)
        elif self.output_type == "mean_norm":
            # scale the tensor signatures so that their expected norm is 1.
            self.output_scale = jnp.stack(
                [
                    jnp.mean(jnp.linalg.norm(t.reshape((self.n_train, -1)), axis=-1))
                    for t in signature
                ]
            )
        elif self.output_type is None:
            self.output_scale = jnp.ones(len(signature))
        elif self.output_type == "standard":
            pass  # implemented above
        else:
            raise ValueError

        scaled_signature = [t / t_scale for t, t_scale in zip(signature, self.output_scale)]
        scaled_signature_flat = jnp.concatenate(
            [t.reshape((len(t), -1)) for t in scaled_signature], axis=-1
        )

        # scale the extra signatures, then flatten them
        scaled_extra_sigs_flat = []
        for extra_signature in extra_signatures:
            scaled_extra_signature = [
                t / t_scale for t, t_scale in zip(extra_signature, self.output_scale)
            ]
            scaled_extra_sigs_flat.append(
                jnp.concatenate([t.reshape((len(t), -1)) for t in scaled_extra_signature], axis=-1)
            )

        return scaled_signature_flat, *scaled_extra_sigs_flat

    def inverse_output(self: Self, signature_flat: jax.Array) -> jax.Array:
        signature = expand_signature(self.D, signature_flat)
        scaled_signature = [t * t_scale for t, t_scale in zip(signature, self.output_scale)]
        return jnp.concatenate([t.reshape((len(t), -1)) for t in scaled_signature], axis=-1)


def map_and_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
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

    pred_signature = expand_signature(D, pred_signature_flat)
    y_signature = expand_signature(D, y)

    loss_per_tensor = jnp.zeros((batch, 0))
    for pred_tensor, y_tensor in zip(pred_signature, y_signature):
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

    pred_signature = expand_signature(D, pred_signature_flat)
    y_signature = expand_signature(D, y)

    loss_per_tensor = jnp.zeros((batch, 0))
    for pred_tensor, y_tensor in zip(pred_signature, y_signature):
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


class SignatureClassifier(eqx.Module):

    net: eqx.nn.Linear

    D: int
    max_k: int
    n_classes: int
    n_in: int

    def __init__(self: Self, D: int, max_k: int, n_classes: int, key: jax.Array) -> None:
        self.D = D
        self.max_k = max_k
        self.n_classes = n_classes
        self.n_in = sum([D**k for k in range(1, max_k + 1)])

        # since the path signature is a universal feature map, there is a linear functional
        # from the path signature to any function.
        self.net = eqx.nn.Linear(self.n_in, self.n_classes, use_bias=False, key=key)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.net(x)


def map_and_loss_classifier(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
    normalizer: Normalizer | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    args:
        model:
        x: input data, shape (batch,steps,D)
        y: output data, shape (batch, sum_k=1^K: D**k)
    """
    assert callable(model)
    pred_y = jax.vmap(model)(x)

    if normalizer is not None:
        pred_y = normalizer.inverse_output(pred_y)
        y = normalizer.inverse_output(y)

    loss = optax.losses.safe_softmax_cross_entropy(pred_y, y)

    return jnp.mean(loss), aux_data


def classifier_misclass_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: eqx.nn.State | None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    Misclassification, so smaller values are better.
    args:
        model:
        x: input data, shape (batch,sum_k=1^K: D**k)
        y: output data, shape (batch,n_classes)
    """
    assert callable(model)
    pred_y = jax.vmap(model)(x)
    return jnp.mean(jnp.argmax(pred_y, axis=1) != jnp.argmax(y, axis=1)), aux_data


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="the name of the dataset", type=str, default=UWAVE)
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("--batch", help="batch size", type=int, default=32)
    parser.add_argument("--n-train", help="number of training points", type=int, default=896)
    parser.add_argument("--n-val", help="number of validation points", type=int, default=896)
    parser.add_argument("--n-test", help="number of testing points", type=int, default=896)
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
    parser.add_argument("-C", help="parameter for C-O normalization", type=float, default=1000)
    parser.add_argument("-a", help="parameter for C-O normalization", type=int, default=1)
    parser.add_argument(
        "--sig-order", help="max order of the tensor signature", type=int, default=3
    )
    parser.add_argument(
        "--steps", help="number of points of the path to use as input", type=int, default=10
    )
    parser.add_argument(
        "--symmetry-breakers",
        help="include input vectors that break the symmetry",
        choices=[None, SYMMETRY_BREAKERS_FULL],
        type=str,
        default=None,
    )
    parser.add_argument(
        "--wandb",
        help="whether to use wandb on this run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb-entity", help="the wandb user", type=str, default="wilson_gregory")
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="signature-time-series"
    )

    return parser.parse_args()


# MAIN
args = handleArgs(sys.argv)
D = 3

key = random.PRNGKey(time.time_ns() if args.seed is None else args.seed)
train_X, train_sig, train_labels, val_X, val_sig, val_labels, test_X, test_sig, test_labels = (
    get_data(
        args.sig_order,
        args.steps,
        args.n_train,
        args.n_val,
        args.n_test,
        args.symmetry_breakers,
    )
)
n_vecs = args.steps + D if args.symmetry_breakers == SYMMETRY_BREAKERS_FULL else args.steps
_, n_classes = train_labels.shape

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    (
        "signax_only",
        1,
        SignaxOnly(args.sig_order),
        Normalizer(D, None, None, args.n_train, args.C, args.a),
    ),
    (
        "tensor_poly",
        1e-4,
        EquivSignature(args.sig_order, n_vecs, 32, 3, False, subkey1),
        Normalizer(D, "gram", "mean_norm", args.n_train, args.C, args.a),
    ),
    (
        "baseline_samewidth",
        5e-3,
        Baseline(D, args.sig_order, n_vecs, 32, 3, subkey4),
        Normalizer(D, "standard", "standard", args.n_train, args.C, args.a),
    ),
    (
        "baseline_sameparams",
        1e-3,
        Baseline(D, args.sig_order, n_vecs, 128, 3, subkey3),
        Normalizer(D, "standard", "standard", args.n_train, args.C, args.a),
    ),
]

results = np.zeros((args.n_trials, len(models_list), 3))
for t in range(args.n_trials):
    for k, (model_name, lr, model, normalizer) in enumerate(models_list):
        # TODO: I need to check that the typical sum of basis elements times the coefficients
        # also has similar standard deviation of norms

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
                map_and_loss,
                model,
                subkey,
                ml.ValLoss(patience=50, max_epochs=args.epochs, verbose=args.verbose),
                args.batch,
                optax.adamw(
                    optax.cosine_onecycle_schedule(args.epochs * steps_per_epoch, lr, div_factor=3)
                ),
                validation_X=val_X_norm,
                validation_Y=val_sig_norm,
                val_map_and_loss=ft.partial(map_and_loss, normalizer=normalizer),
                save_model=save_model,
                is_wandb=args.wandb,
            )

            if args.wandb:
                wandb.finish()

            if args.save_model is not None:
                ml.save(save_model, trained_model)

        key, subkey1, subkey2, subkey3 = random.split(key, num=4)
        results[t, k, 0], pred_train_sig = ml.map_loss_in_batches(
            ft.partial(map_and_loss_return_map, normalizer=normalizer),
            trained_model,
            train_X_norm,
            train_sig_norm,
            args.batch,
            subkey1,
            devices=jax.devices(),
            return_map=True,
        )
        results[t, k, 1], pred_val_sig = ml.map_loss_in_batches(
            ft.partial(map_and_loss_return_map, normalizer=normalizer),
            trained_model,
            val_X_norm,
            val_sig_norm,
            args.batch,
            subkey1,
            devices=jax.devices(),
            return_map=True,
        )
        results[t, k, 2], pred_test_sig = ml.map_loss_in_batches(
            ft.partial(map_and_loss_return_map, normalizer=normalizer),
            trained_model,
            test_X_norm,
            test_sig_norm,
            args.batch,
            subkey1,
            devices=jax.devices(),
            return_map=True,
        )
        print(f"{t},{model_name}: {results[t,k,2]}")

        key, subkey = random.split(key)
        signature_classifier = SignatureClassifier(D, args.sig_order, n_classes, subkey)

        steps_per_epoch = int(args.n_train / args.batch)
        key, subkey = random.split(key)
        trained_model, _, _, _ = ml.train(
            pred_train_sig,
            train_labels,
            map_and_loss_classifier,
            signature_classifier,
            subkey,
            # ml.EpochStop(epochs=100, verbose=2),
            ml.ValLoss(patience=50, max_epochs=args.epochs, verbose=args.verbose),
            args.batch,
            optax.adamw(
                optax.cosine_onecycle_schedule(args.epochs * steps_per_epoch, 1e-3, div_factor=3)
            ),
            validation_X=pred_val_sig,
            validation_Y=val_labels,
            val_map_and_loss=classifier_misclass_loss,
        )

        key, subkey = random.split(key)
        val_loss = ml.map_loss_in_batches(
            classifier_misclass_loss,
            trained_model,
            pred_val_sig,
            val_labels,
            args.batch,
            subkey,
            devices=jax.devices(),
        )
        print("Val accuracy:", 1 - val_loss)


print("Test Errors")
for k, (model_name, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
