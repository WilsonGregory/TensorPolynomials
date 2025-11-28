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


def voigt_to_full(D: int, rows: ArrayLike) -> jax.Array:
    """
    convert voigt notation tensors ['C11', 'C22', 'C33', 'C23', 'C13', 'C12'] to full
    tensors.
    args:
        rows (jnp.Array): (batch,6)
    """
    assert D == 3

    def form_tensor(voigt_elems):
        base = jnp.zeros((D, D))
        # I don't think there is a way around using at and set
        base = base.at[jnp.triu_indices(D, k=1)].set(jnp.flip(voigt_elems[D:]))
        return jnp.diag(voigt_elems[:D]) + jnp.array(base) + jnp.array(base).T

    return jax.vmap(form_tensor)(rows)


def load_one_file(D: int, file_path: str, num: int) -> tuple[jax.Array, jax.Array]:
    # tensors are saved in Voigt notation, so we convert them to full representation
    rows = jnp.array(np.loadtxt(file_path, delimiter=",", skiprows=1))[:num]
    return voigt_to_full(D, rows[:, :6]), voigt_to_full(D, rows[:, 6:])


# data comes from a csv filepath
def get_data(
    D: int,
    dir_path: str,
    train_file: str,
    test_file: str,
    n_train: int,
    n_val: int,
    n_test: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    train_X, train_y = load_one_file(D, dir_path + train_file, n_train)
    test_X, test_y = load_one_file(D, dir_path + test_file, n_val + n_test)

    # now split test, val into two data sets
    val_X = test_X[:n_val]
    val_y = test_y[:n_val]

    test_X = test_X[n_val:]
    test_y = test_y[n_val:]

    return train_X, train_y, val_X, val_y, test_X, test_y


def normalize_data(
    D: int,
    train_X: jax.Array,
    train_y: jax.Array,
    val_X: jax.Array,
    val_y: jax.Array,
    test_X: jax.Array,
    test_y: jax.Array,
    normalize_type: str,
    global_y_std: jax.Array,
):
    if normalize_type == "componentwise":
        mean_X = jnp.mean(train_X, axis=0, keepdims=True)  # (1,D,D)
        mean_y = jnp.mean(train_y, axis=0, keepdims=True)
        std_X = jnp.std(train_X, axis=0, keepdims=True)
        std_y = jnp.std(train_y, axis=0, keepdims=True)

        scaler = std_y / global_y_std

    elif normalize_type == "tensor":
        # Subtract a multiple of the identity so that the mean diagonal value is 0
        mean_X = jnp.mean(jnp.diag(jnp.mean(train_X, axis=0))) * jnp.eye(D)[None]
        mean_y = jnp.mean(jnp.diag(jnp.mean(train_y, axis=0))) * jnp.eye(D)[None]
        std_X = jnp.std(jnp.linalg.norm(train_X - mean_X, axis=(1, 2)))
        std_y = jnp.std(jnp.linalg.norm(train_y - mean_y, axis=(1, 2)))

        scaler = std_y / global_y_std  # this will be 1

    elif normalize_type == "eigenvalues_perm":
        eigvals_X = jnp.linalg.eigvalsh(train_X)
        mean_X = jnp.mean(eigvals_X) * jnp.eye(D)[None]  # works in every basis
        eigvals_y = jnp.linalg.eigvalsh(train_y)
        mean_y = jnp.mean(eigvals_y) * jnp.eye(D)[None]
        std_X = jnp.std(eigvals_X)
        std_y = jnp.std(eigvals_y)

        scaler = std_y / global_y_std

    train_X = train_X - mean_X
    val_X = val_X - mean_X
    test_X = test_X - mean_X

    train_y = train_y - mean_y
    val_y = val_y - mean_y
    test_y = test_y - mean_y

    train_X = train_X / std_X
    val_X = val_X / std_X
    test_X = test_X / std_X

    train_y = train_y / std_y
    val_y = val_y / std_y
    test_y = test_y / std_y

    return train_X, train_y, val_X, val_y, test_X, test_y, scaler


def augment_data(
    D: int,
    data: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    aug_multiplier: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data
    batch = len(train_X)

    train_X_aug = jnp.full((aug_multiplier,) + train_X.shape, train_X[None])
    train_X_aug = train_X_aug.reshape((batch * aug_multiplier, D, D))

    train_Y_aug = jnp.full((aug_multiplier,) + train_Y.shape, train_Y[None])
    train_Y_aug = train_Y_aug.reshape((batch * aug_multiplier, D, D))

    key, subkey = random.split(key)
    Qs = random.orthogonal(subkey, D, (batch * aug_multiplier,))
    train_X_aug_rot = jnp.einsum(
        "...ij,...kl,...jl->...ik", Qs, Qs, train_X_aug, precision=jax.lax.Precision.HIGHEST
    )
    train_Y_aug_rot = jnp.einsum(
        "...ij,...kl,...jl->...ik", Qs, Qs, train_Y_aug, precision=jax.lax.Precision.HIGHEST
    )

    return train_X_aug_rot, train_Y_aug_rot, val_X, val_Y, test_X, test_Y


class CountableModule(eqx.Module):
    """
    An equinox module with the function count_params defined.
    """

    def count_params(self: Self) -> int:
        return sum(
            [
                0 if x is None else x.size
                for x in eqx.filter(jax.tree_util.tree_leaves(self), eqx.is_array)
            ]
        )


class BaselineMLP(CountableModule):
    D: int
    net: eqx.nn.MLP

    def __init__(self: Self, D: int, width: int, num_hidden_layers: int, key: jax.Array):
        """
        Baseline model which is just an mlp.
        """
        self.D = D
        self.net = eqx.nn.MLP(D**2, D**2, width, num_hidden_layers, jax.nn.gelu, key=key)

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        return self.net(x.reshape(-1)).reshape((D,) * 2)


class TwoTensorMap(CountableModule):
    D: int
    net: eqx.nn.MLP

    def __init__(self: Self, D: int, width: int, n_hidden_layers: int, key: jax.Array):
        """
        Constructor for model whose input is a symmetric 2-tensor and output is a
        symmetric 2-tensor. It performs an eigenvalue decomposition, maps the eigenvalues with
        an MLP, then reconstructs the matrix.
        args:
            D (int): dimension of the tensor
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        self.D = D
        self.net = eqx.nn.MLP(self.D, self.D, width, n_hidden_layers, jax.nn.gelu, key=key)

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        """
        args:
            x (jnp.array): (d,d) array,
        """
        eigvals, eigvecs = jnp.linalg.eigh(x)
        return eigvecs @ jnp.diag(self.net(eigvals)) @ eigvecs.T


class TwoTensorMapPermEquivariant(CountableModule):
    D: int
    equiv_layers: list[ml.PermEquivariantLayer]

    def __init__(self: Self, D: int, width: int, n_hidden_layers: int, key: ArrayLike):
        """
        Constructor for model whose input is a symmetric 2-tensor and output is a
        symmetric 2-tensor. It performs an eigenvalue decomposition, maps the eigenvalues, then
        reconstructs the matrix. In this version, the function that maps the eigenvalues is
        permutation equivariant, as the theory suggests.
        args:
            D (int): dimension of the tensor
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        self.D = D

        key, subkey = random.split(key)
        self.equiv_layers = [ml.PermEquivariantLayer({1: 1}, {1: width}, D, subkey)]
        for _ in range(n_hidden_layers - 1):
            key, subkey = random.split(key)
            self.equiv_layers.append(ml.PermEquivariantLayer({1: width}, {1: width}, D, subkey))

        key, subkey = random.split(key)
        self.equiv_layers.append(ml.PermEquivariantLayer({1: width}, {1: 1}, D, subkey))

    def __call__(self: Self, x: jax.Array) -> jax.Array:
        """
        args:
            x (jnp.array): (d,d) array,
        """
        eigvals, eigvecs = jnp.linalg.eigh(x)
        evals_dict = {1: eigvals[None]}  # (1,D)
        for layer in self.equiv_layers[:-1]:
            evals_dict = {k: jax.nn.gelu(tensor) for k, tensor in layer(evals_dict).items()}

        eigvals = self.equiv_layers[-1](evals_dict)[1].reshape((3,))

        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    def count_params(self: Self) -> int:
        return sum(x.count_params() for x in self.equiv_layers)


def map_and_loss(
    model: eqx.Module,
    x: jax.Array,
    y: jax.Array,
    aux_data: Optional[eqx.nn.State],
    scaler: Optional[Union[float, jax.Array]] = None,
):
    """
    Map x using the model,
    args:
        model (functional): function on a single input, will be vmapped
        x (jnp.array): input data, shape (batch,d,d)
        y (jnp.array): output data, the sparse vector, shape (batch,d,d)
        aux_data (): optional aux data used for BatchNorm and the like
        scaler ():
    """
    assert callable(model)
    pred_y = jax.vmap(model)(x)
    if scaler is not None:
        pred_y = pred_y * scaler
        y = y * scaler

    # use mse, mean over batch, sum over tensor components
    return jnp.mean(jnp.linalg.matrix_norm(pred_y - y) ** 2), aux_data


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", help="the folder containing neohookean_train.csv and neohookean_val.csv", type=str
    )
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("--batch", help="batch size", type=int, default=256)
    parser.add_argument("--n-train", help="number of training points", type=int, default=20000)
    parser.add_argument("--n-val", help="number of validation points", type=int, default=4000)
    parser.add_argument("--n-test", help="number of testing points", type=int, default=4000)
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
        "--wandb-project", help="the wandb project", type=str, default="tensor-polynomials"
    )

    return parser.parse_args()


# Main
args = handleArgs(sys.argv)
key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)
D = 3

# define training params
width = 23
n_hidden_layers = 3

data = get_data(
    D,
    args.data,
    "neohookean_train.csv",
    "neohookean_val.csv",
    args.n_train,
    args.n_val,
    args.n_test,
)
global_y_std = jnp.std(data[1])

key, subkey1, subkey2, subkey3, subkey4 = random.split(key, num=5)
models_list = [
    (
        "TwoTensorMapPermEquivariant23",
        1e-3,  # 1e-3 for 20k, 2e-3 for 5k
        TwoTensorMapPermEquivariant(D, width, n_hidden_layers, subkey1),
        1,
        "eigenvalues_perm",
    ),
    # having trouble reproducing results with the non-permutation equivariant model
    # this model was already "canonicalized" because eigh returns ordered eigenvalues.
    ("TwoTensorMap32", 3e-3, TwoTensorMap(D, 32, n_hidden_layers, subkey2), 1, "eigenvalues_perm"),
    ("BaselineMLP32", 3e-3, BaselineMLP(D, 32, n_hidden_layers, subkey3), 1, "componentwise"),
    ("BaselineMLP32_aug4", 3e-3, BaselineMLP(D, 32, n_hidden_layers, subkey4), 4, "tensor"),
]

for model_name, _, model, _, _ in models_list:
    print(f"{model_name} params: {model.count_params():,}")

results = np.zeros((args.n_trials, len(models_list), 3))
for t in range(args.n_trials):
    for k, (model_name, lr, model, aug_multiplier, normalize_type) in enumerate(models_list):
        key, subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, num=6)

        name = f"materials_{model_name}_t{t}_lr{lr}_e{args.epochs}_ntrain{args.n_train}"
        save_model = f"{args.save_model}/{name}.eqx" if args.save_model is not None else None

        if aug_multiplier > 1:
            aug_data = augment_data(D, data, aug_multiplier, subkey1)
        else:
            aug_data = data

        train_X, train_y, val_X, val_y, test_X, test_y, scaler = normalize_data(
            D, *aug_data, normalize_type, global_y_std
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
                        "normalize_type": normalize_type,
                    }
                )

            steps_per_epoch = int(len(train_X) / args.batch)
            trained_model, _, _, _ = ml.train(
                train_X,
                train_y,
                map_and_loss,
                model,
                subkey2,
                ml.EpochStop(epochs=args.epochs, verbose=args.verbose),
                args.batch,
                optax.adamw(
                    optax.cosine_onecycle_schedule(args.epochs * steps_per_epoch, lr, div_factor=3)
                ),
                validation_X=val_X,
                validation_Y=val_y,
                val_map_and_loss=ft.partial(map_and_loss, scaler=scaler),  # correct scaler?
                save_model=save_model,
                is_wandb=args.wandb,
            )

            if args.wandb:
                wandb.finish()

            if args.save_model is not None:
                ml.save(save_model, trained_model)

        results[t, k, 0] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, scaler=scaler),
            trained_model,
            train_X,
            train_y,
            args.batch,
            subkey3,
            devices=jax.devices(),
        )
        results[t, k, 1] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, scaler=scaler),
            trained_model,
            val_X,
            val_y,
            args.batch,
            subkey4,
            devices=jax.devices(),
        )
        results[t, k, 2] = ml.map_loss_in_batches(
            ft.partial(map_and_loss, scaler=scaler),
            trained_model,
            test_X,
            test_y,
            args.batch,
            subkey5,
            devices=jax.devices(),
        )
        print(f"{t},{model_name}: {results[t,k,2]}")

print(results)

print("Test Errors")
for k, (model_name, _, _, _, _) in enumerate(models_list):
    print(f"{model_name}: {jnp.mean(results[:, k, 2]):.3e} +- {jnp.std(results[:, k, 2]):.3e}")
