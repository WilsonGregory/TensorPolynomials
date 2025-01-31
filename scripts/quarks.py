import time
import numpy as np
from typing_extensions import Optional, Self, Union

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import equinox as eqx
import optax

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils

EPS = 1e-5


def print_hist(data_X, data_Y, name, bins=17):
    data_X_0 = jnp.sum(jnp.linalg.norm(data_X[data_Y == 0], axis=2) > EPS, axis=1)
    data_X_0_pct = jnp.around(
        jnp.histogram(data_X_0, bins=bins, range=(0.0, bins * 10.0))[0] / len(data_X_0), 3
    )
    print(f"{name}_X_0", data_X_0_pct)

    data_X_1 = jnp.sum(jnp.linalg.norm(data_X[data_Y == 1], axis=2) > EPS, axis=1)
    data_X_1_pct = jnp.around(
        jnp.histogram(data_X_1, bins=bins, range=(0.0, bins * 10.0))[0] / len(data_X_1),
        3,
    )
    print(f"{name}_X_1", data_X_1_pct)


def read_one_npz(
    filename: str,
    n_train: Optional[int],
    n_val: Optional[int],
    n_test: Optional[int],
    n_points: Optional[int] = 165,
    normalize=False,
    add_time_vector=True,
    add_xy_plane=True,
    print_dist=False,
):
    # n_points is the max number of n_points to use, they are zero padded up to 200, but it looks
    # like the max actually in the data set is 165, so make that the default.

    with np.load(filename) as data:
        train_X = jnp.array(data["kinematics_train"][:n_train, :n_points])  # (n_train, n_points, 4)
        train_Y = jnp.array(data["labels_train"][:n_train])  # (n_train,)

        val_X = jnp.array(data["kinematics_val"][:n_val, :n_points])  # (n_val, n_points, 4)
        val_Y = jnp.array(data["labels_val"][:n_val])  # (n_val,)

        test_X = jnp.array(data["kinematics_test"][:n_test, :n_points])  # (n_test, n_points, 4)
        test_Y = jnp.array(data["labels_test"][:n_test])  # (n_test,)

        n_train = len(train_X)
        n_val = len(val_X)
        n_test = len(test_X)

        if print_dist:
            print_hist(train_X, train_Y, "train")
            print_hist(test_X, test_Y, "test")

        if normalize:
            stdev = jnp.std(train_X)
            train_X = train_X / stdev
            val_X = val_X / stdev
            test_X = test_X / stdev

        if add_xy_plane:
            x_plus = jnp.array([0, 0, 0, 1]).reshape((1, 1, 4))
            x_minus = jnp.array([0, 0, 0, -1]).reshape((1, 1, 4))
            x_pm = jnp.concatenate([x_plus, x_minus], axis=1)
            train_X = jnp.concatenate([x_pm.repeat(n_train, axis=0), train_X], axis=1)
            val_X = jnp.concatenate([x_pm.repeat(n_val, axis=0), val_X], axis=1)
            test_X = jnp.concatenate([x_pm.repeat(n_test, axis=0), test_X], axis=1)

        if add_time_vector:
            time_vec = jnp.array([1, 0, 0, 0]).reshape((1, 1, 4))
            train_X = jnp.concatenate([time_vec.repeat(n_train, axis=0), train_X], axis=1)
            val_X = jnp.concatenate([time_vec.repeat(n_val, axis=0), val_X], axis=1)
            test_X = jnp.concatenate([time_vec.repeat(n_test, axis=0), test_X], axis=1)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y


class LorentzScalarsAreUniversal(eqx.Module):
    layers: list[eqx.Module]

    def __init__(self: Self, n_vecs: int, width: int, num_hidden_layers: int, key: ArrayLike):
        """
        Constructor for SparseVectorHunter, parameterizes full function from vectors to 2-tensor.
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        in_features = n_vecs + (n_vecs * (n_vecs - 1) // 2)

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Linear(in_features, width, key=subkey)]
        for _ in range(num_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(width, width, key=subkey))

        key, subkey = random.split(key)
        self.layers.append(eqx.nn.Linear(width, 1, key=subkey))

    def __call__(self: Self, S: ArrayLike):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        n_vecs, _ = S.shape  # (200,4)
        metric = jnp.diag(jnp.array([1, -1, -1, -1]))  # the lorentz metric
        X = (S @ metric @ S.T)[jnp.triu_indices(n_vecs)].reshape(-1)
        assert X.shape == (n_vecs + n_vecs * (n_vecs - 1) // 2,)

        for layer in self.layers[:-1]:
            X = jax.nn.relu(layer(X))

        return self.layers[-1](X)[0]


class LorentzPermInvariant(eqx.Module):
    perm_layers: list[ml.PermEquivariantLayer]
    layers: list[eqx.nn.Linear]
    n_particles: int
    n_added_vecs: int

    def __init__(
        self: Self,
        n_particles: int,
        n_added_vecs: int,
        equiv_width: int,
        inv_width: int,
        n_hidden_perm_equiv: int,
        n_hidden_perm_inv: int,
        key: ArrayLike,
    ):
        """
        Constructor for Lorentz equivariant, permutation invariant network based on just scalars.
        args:
            n_vecs (int): number of input vectors
            width (int): width of the NN layers
            n_hidden_perm_equiv (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        self.n_particles = n_particles
        self.n_added_vecs = n_added_vecs

        self.perm_layers = []
        input_keys = {
            0: n_added_vecs + n_added_vecs * (n_added_vecs - 1) // 2,
            1: n_added_vecs,
            2: 1,
        }
        if n_hidden_perm_equiv > 0:
            # this basis maps a perm invariant vector to a perm invariant vector
            output_keys = {0: equiv_width, 1: equiv_width, 2: equiv_width}

            key, subkey = random.split(key)
            self.perm_layers.append(
                ml.PermEquivariantLayer(input_keys, output_keys, n_particles, subkey)
            )
            input_keys = output_keys

            for _ in range(n_hidden_perm_equiv - 1):
                key, subkey = random.split(key)
                self.perm_layers.append(
                    ml.PermEquivariantLayer(input_keys, output_keys, n_particles, subkey)
                )

        key, subkey = random.split(key)
        self.perm_layers.append(
            ml.PermEquivariantLayer(input_keys, {0: inv_width}, n_particles, subkey)
        )

        self.layers = []
        for _ in range(n_hidden_perm_inv):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(inv_width, inv_width, key=subkey))

        key, subkey = random.split(key)
        self.layers.append(eqx.nn.Linear(inv_width, 1, key=subkey))

    def __call__(self: Self, S: ArrayLike):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        added_vecs = S[: self.n_added_vecs]
        particles = S[self.n_added_vecs :]

        metric = jnp.diag(jnp.array([1, -1, -1, -1]))  # the lorentz metric

        particles_X = (particles @ metric @ particles.T)[None]
        assert particles_X.shape == (1, self.n_particles, self.n_particles)

        vecs_particles_X = added_vecs @ metric @ particles.T  # (n_added_vecs,n_particles)
        assert vecs_particles_X.shape == (self.n_added_vecs, self.n_particles)

        vecs_vecs = (added_vecs @ metric @ added_vecs.T)[
            jnp.triu_indices(self.n_added_vecs)
        ].reshape(-1)
        assert vecs_vecs.shape == (
            self.n_added_vecs + self.n_added_vecs * (self.n_added_vecs - 1) // 2,
        )

        X = {0: vecs_vecs, 1: vecs_particles_X, 2: particles_X}

        for layer in self.perm_layers:
            X = {k: jax.nn.gelu(tensor) for k, tensor in layer(X).items()}

        X = X[0]  # last perm layer maps to all scalars

        for layer in self.layers[:-1]:
            X = jax.nn.gelu(layer(X))

        return self.layers[-1](X)[0]

    def count_params(self: Self):
        return sum([layer.count_params() for layer in self.perm_layers]) + sum(
            [x.size for x in jax.tree_util.tree_leaves(eqx.filter(self.layers, eqx.is_array))]
        )


def map_and_loss(model, x, y, batch_state):
    pred_y = jax.vmap(model)(x)

    log_p = jax.nn.log_softmax(pred_y)
    log_not_p = jax.nn.log_softmax(-pred_y)
    # log_p = jax.nn.log_sigmoid(pred_y)
    # log_not_p = jax.nn.log_sigmoid(-pred_y)
    return jnp.mean(-y * log_p - (1.0 - y) * log_not_p), batch_state


def map_and_accuracy(model, x, y, batch_state):
    pred_y = jnp.around(jax.nn.sigmoid(jax.vmap(model)(x)))  # map, then threshold to 0 or 1
    return jnp.mean((pred_y - y) == 0), batch_state


# data params

foo = utils.PermInvariantTensor.get(4, 10, ((0, 1), (2, 3)))
filename = "/data/wgregor4/quarks/toptagging_full.npz"
n_train = 32768
n_val = 2048
n_test = None
n_points = 80  # 165 is the max number of vectors
normalize = True
add_time_vector = True
add_xy_plane = True

n_added_vecs = 1 * add_time_vector + 2 * add_xy_plane

train_X, train_Y, val_X, val_Y, test_X, test_Y = read_one_npz(
    filename, n_train, n_val, n_test, n_points, normalize, add_time_vector, add_xy_plane
)

# training params
batch_size = 128
verbose = 2
epochs = 20
lr = 1e-2

key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
model = LorentzPermInvariant(train_X.shape[1] - n_added_vecs, n_added_vecs, 8, 128, 1, 3, subkey)
# model = LorentzScalarsAreUniversal(train_X.shape[1], 128, 5, subkey)
print(f"{model.count_params():,} params")

save_model = (
    f"/data/wgregor4/runs/quarks/lorentz_perm_equiv_n{n_train}_b{batch_size}_e{epochs}_lr{lr}.eqx"
)
load_model = None

# save_model = None
# load_model = save_model

devices = jax.devices()

if load_model:
    trained_model = ml.load(load_model, model)
else:
    steps_per_epoch = int(len(train_X) / batch_size)
    key, subkey = random.split(key)
    trained_model, _, _, _ = ml.train(
        train_X,
        train_Y,
        map_and_loss,
        model,
        subkey,
        # ml.ValLoss(patience=5, verbose=verbose),
        ml.EpochStop(epochs=epochs, verbose=verbose),
        batch_size,
        optax.adam(optax.exponential_decay(lr, steps_per_epoch, 0.99)),
        val_X,
        val_Y,
        save_model,
        devices=devices,
    )

    if save_model:
        ml.save(save_model, trained_model)

key, subkey1, subkey2 = random.split(key, 3)
print(
    "train:",
    ml.map_loss_in_batches(
        map_and_accuracy, trained_model, train_X, train_Y, batch_size, subkey1, devices=devices
    ),
)
print(
    "test:",
    ml.map_loss_in_batches(
        map_and_accuracy, trained_model, test_X, test_Y, batch_size, subkey2, devices=devices
    ),
)
