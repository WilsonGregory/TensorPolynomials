import sys
import time
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


def load_one_file(D: int, file_path: str, num: int) -> tuple[jax.Array]:
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
    normalize: bool = True,
) -> tuple[jax.Array]:
    train_X, train_y = load_one_file(D, dir_path + train_file, n_train)
    test_X, test_y = load_one_file(D, dir_path + test_file, n_val + n_test)

    if normalize:
        # Subtract a multiple of the identity so that the mean diagonal value is 0
        mean_X_scalar = jnp.mean(jnp.diag(jnp.mean(train_X, axis=0)))
        train_X = train_X - mean_X_scalar * jnp.eye(D)
        test_X = test_X - mean_X_scalar * jnp.eye(D)

        mean_y_scalar = jnp.mean(jnp.diag(jnp.mean(train_y, axis=0)))
        train_y = train_y - mean_y_scalar * jnp.eye(D)
        test_y = test_y - mean_y_scalar * jnp.eye(D)

        # Divide by the standard deviation so that the std component is 1
        std_X_scalar = jnp.std(train_X)
        train_X = train_X / std_X_scalar
        test_X = test_X / std_X_scalar

        std_y_scalar = jnp.std(train_y)
        train_y = train_y / std_y_scalar
        test_y = test_y / std_y_scalar

    # now split test, val into two data sets
    val_X = test_X[:n_val]
    val_y = test_y[:n_val]

    test_X = test_X[n_val:]
    test_y = test_y[n_val:]

    return train_X, train_y, val_X, val_y, test_X, test_y


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

    def __init__(self: Self, D: int, width: int, num_hidden_layers: int, key: ArrayLike):
        """
        Baseline model which is just an mlp.
        """
        self.D = D
        self.net = eqx.nn.MLP(D**2, D**2, width, num_hidden_layers, key=key)

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        return self.net(x.reshape(-1)).reshape((D,) * 2)


class TwoTensorMap(CountableModule):
    D: int
    net: eqx.nn.MLP

    def __init__(self: Self, D: int, width: int, n_hidden_layers: int, key: ArrayLike):
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
        self.net = eqx.nn.MLP(self.D, self.D, width, n_hidden_layers, key=key)

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

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        """
        args:
            x (jnp.array): (d,d) array,
        """
        eigvals, eigvecs = jnp.linalg.eigh(x)
        evals_dict = {1: eigvals[None]}  # (1,D)
        for layer in self.equiv_layers[:-1]:
            evals_dict = {k: jax.nn.relu(tensor) for k, tensor in layer(evals_dict).items()}

        eigvals = self.equiv_layers[-1](evals_dict)[1].reshape((3,))

        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T

    def count_params(self: Self) -> int:
        return sum(x.count_params() for x in self.equiv_layers)


def map_and_loss(model, x, y, aux_data):
    """
    Map x using the model,
    args:
        model (functional): function on a single input, will be vmapped
        x (jnp.array): input data, shape (batch,n,d)
        y (jnp.array): output data, the sparse vector, shape (batch,n)
    """
    pred_y = jax.vmap(model)(x)
    # use mse, mean over batch, sum over tensor components
    return jnp.mean(jnp.linalg.matrix_norm(pred_y - y)), aux_data


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data", help="the folder containing neohookean_train.csv and neohookean_val.csv", type=str
    )
    parser.add_argument("-e", "--epochs", help="number of epochs to run", type=int, default=50)
    parser.add_argument("--batch", help="batch size", type=int, default=256)
    parser.add_argument("--n_train", help="number of training points", type=int, default=20000)
    parser.add_argument("--n_val", help="number of validation points", type=int, default=4000)
    parser.add_argument("--n_test", help="number of testing points", type=int, default=4000)
    parser.add_argument("-t", "--n_trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("--seed", help="the random number seed", type=int, default=None)
    parser.add_argument(
        "-s", "--save_model", help="file name to save the params", type=str, default=None
    )
    parser.add_argument(
        "-l", "--load_model", help="file name to load params from", type=str, default=None
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

train_X, train_y, val_X, val_y, test_X, test_y = get_data(
    D,
    args.data,
    "neohookean_train.csv",
    "neohookean_val.csv",
    args.n_train,
    args.n_val,
    args.n_test,
)

key, subkey1, subkey2, subkey3 = random.split(key, 4)
models_list = [
    (
        "TwoTensorMapPermEquivariant23",
        7e-4,
        TwoTensorMapPermEquivariant(D, width, n_hidden_layers, subkey1),
    ),
    ("TwoTensorMap32", 1e-3, TwoTensorMap(D, 32, n_hidden_layers, subkey2)),
    ("BaselineMLP32", 1e-3, BaselineMLP(D, 32, n_hidden_layers, subkey3)),
]

for model_name, lr, model in models_list:
    print(f"{model_name} params: {model.count_params():,}")

results = np.zeros((args.n_trials, len(models_list), 2))
for t in range(args.n_trials):
    for k, (model_name, lr, model) in enumerate(models_list):

        name = f"materials_{model_name}_t{t}_lr{lr}_bs{args.batch}"
        print(name)

        if args.load_model:
            trained_model = ml.load(args.load_model, model)
        else:
            if args.wandb:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,  # what is this?
                    name=name,
                    settings=wandb.Settings(start_method="fork"),
                )
                wandb.config.update(args)
                wandb.config.update({"trial": t, "model_name": model_name, "lr": lr})

            steps_per_epoch = int(args.n_train / args.batch)
            key, subkey = random.split(key)
            trained_model, _, _, _ = ml.train(
                train_X,
                train_y,
                map_and_loss,
                model,
                subkey,
                ml.EpochStop(epochs=args.epochs, verbose=args.verbose),
                args.batch,
                optax.adam(optax.exponential_decay(lr, steps_per_epoch, 0.995)),
                validation_X=val_X,
                validation_Y=val_y,
                save_model=args.save_model,
                is_wandb=args.wandb,
            )

            if args.wandb:
                wandb.finish()

            if args.save_model is not None:
                ml.save(args.save_model, trained_model)

        results[t, k, 0] = map_and_loss(trained_model, train_X, train_y, None)[0]
        results[t, k, 1] = map_and_loss(trained_model, test_X, test_y, None)[0]
        print(f"{t},{model_name}: {results[t,k,1]}")

print(results)
print("mean over trials", jnp.mean(results, axis=0))
