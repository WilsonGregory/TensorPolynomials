import time
import numpy as np
from typing_extensions import Self

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
    n_test: int,
    normalize: bool = True,
) -> tuple[jax.Array]:
    train_X, train_y = load_one_file(D, dir_path + train_file, n_train)
    test_X, test_y = load_one_file(D, dir_path + test_file, n_test)

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

    return train_X, train_y, test_X, test_y


class BaselineMLP(eqx.Module):
    D: int
    mlp: eqx.nn.MLP

    def __init__(self: Self, D: int, width: int, num_hidden_layers: int, key: ArrayLike):
        """
        Baseline model which is just an mlp.
        """
        self.D = D
        self.mlp = eqx.nn.MLP(D**2, D**2, width, num_hidden_layers, key=key)

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        return self.mlp(x.reshape(-1)).reshape((D,) * 2)


class SVHTwoTensor(eqx.Module):
    D: int
    net: eqx.nn.MLP

    def __init__(self: Self, D: int, width: int, num_hidden_layers: int, key: ArrayLike):
        """
        Constructor for SparseVectorHunter whose input is a symmetric 2-tensor and output is a
        symmetric 2-tensor.
        args:
            D (int): dimension of the tensor
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        self.D = D
        self.net = eqx.nn.MLP(self.D + 1, "scalar", width, num_hidden_layers, key=key)

    def __call__(self: Self, x: ArrayLike) -> jax.Array:
        """
        args:
            x (jnp.array): (d,d) array,
        """
        eigvals, eigvecs = jnp.linalg.eigh(x)
        # construct (d,d+1) where the ith row is [lambda_1, lambda_2, lambda_3, lambda_i]
        extended_eigvals = jnp.concatenate(
            [jnp.full((3, 3), eigvals), eigvals.reshape((3, 1))], axis=1
        )
        out = jax.vmap(self.net)(extended_eigvals)

        return eigvecs @ jnp.diag(out) @ eigvecs.T


class SVHTwoTensorPermEquivariant(eqx.Module):
    D: int
    equiv_layers: list[ml.PermEquivariantLayer]

    def __init__(self: Self, D: int, width: int, n_hidden_layers: int, key: ArrayLike):
        """
        Version of SVH with symmetric 2-tensor inputs/outputs where the polynomial of eigenvalues
        is explicitly symmetric in the first three arguments.
        """
        self.D = D

        key, subkey = random.split(key)
        self.equiv_layers = [ml.PermEquivariantLayer({1: 1}, {1: width}, D, subkey)]
        for _ in range(n_hidden_layers):
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

        eigvals = self.equiv_layers[-1](evals_dict)[1][0]  # select the first (only) channel

        return eigvecs @ jnp.diag(eigvals) @ eigvecs.T


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


# Main
key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
train = True
save = False
save_dir = "../runs/"
table_print = True

# define data params
D = 3
n_train = 5000
n_test = 4000

# define training params
width = 23
num_hidden_layers = 3
batch_size = 256
trials = 1
verbose = 2

train_X, train_y, test_X, test_y = get_data(
    D,
    "/data/wgregor4/tfenn/neohookean/",
    "neohookean_train.csv",
    "neohookean_val.csv",
    n_train,
    n_test,
)

key, subkey1, subkey2, subkey3 = random.split(key, 4)
models_list = [
    (
        "SVHTwoTensorPermEquivariant",
        1e-3,
        SVHTwoTensorPermEquivariant(D, width, num_hidden_layers, subkey2),
    ),
    ("SVHTwoTensor", 1e-3, SVHTwoTensor(D, width, num_hidden_layers, subkey1)),
    ("BaselineMLP", 1e-3, BaselineMLP(D, 2 * width, num_hidden_layers, subkey3)),
]

for model_name, lr, model in models_list:
    num_params = sum(
        [
            0 if x is None else x.size
            for x in eqx.filter(jax.tree_util.tree_leaves(model), eqx.is_array)
        ]
    )
    print(f"{model_name} params: {num_params:,}")

results = np.zeros((trials, len(models_list), 2))
if train:
    for t in range(trials):
        for k, (model_name, lr, model) in enumerate(models_list):
            steps_per_epoch = int(n_train / batch_size)
            key, subkey = random.split(key)
            trained_model, _, _, _ = ml.train(
                train_X,
                train_y,
                map_and_loss,
                model,
                subkey,
                ml.EpochStop(epochs=20, verbose=verbose),
                batch_size,
                optax.adam(optax.exponential_decay(lr, int(n_train / batch_size), 0.995)),
            )

            results[t, k, 0] = map_and_loss(trained_model, train_X, train_y, None)[0]
            results[t, k, 1] = map_and_loss(trained_model, test_X, test_y, None)[0]
            print(f"{t},{model_name}: {results[t,k,1]}")

print(results)
print("mean over trials", jnp.mean(results, axis=0))
