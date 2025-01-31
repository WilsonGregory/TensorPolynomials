import time
import math
import functools
from typing_extensions import Any, Callable, Optional, Union, Self

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, ArrayLike, PyTree
import equinox as eqx
import optax

import tensorpolynomials.utils as utils


class GeneralLinear(eqx.Module):
    """
    Performs a linear (affine) map, using a specified basis rather than the usual linear basis.
    The basis for both the linear map and the bias must be specified.
    """

    basis: Array
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    bias_basis: Array
    weights: Array
    bias: Array

    def __init__(
        self,
        basis: Array,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        bias_basis: Array = None,
        key=None,
    ):
        self.basis = basis
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias_basis = bias_basis

        wkey, bkey = random.split(key)

        lim = 1 / math.sqrt(in_features * len(self.basis))
        self.weights = random.uniform(
            wkey,
            shape=(out_features, in_features, len(self.basis)),
            minval=-lim,
            maxval=lim,
        )

        if self.use_bias:
            self.bias = random.uniform(
                bkey,
                shape=(out_features, len(self.bias_basis)),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

    def __call__(self, x: Array, key=None):
        # x.shape: (in,tensor_in)
        in_k = x.ndim - 1
        out_k = self.basis.ndim - in_k - 1
        # (out,in,len_basis), (len_basis,tensor_in_out) -> (out,in,tensor_in_out)
        tensor_map = jnp.einsum("abc,c...->ab...", self.weights, jax.lax.stop_gradient(self.basis))

        in_k_str = utils.LETTERS[2 : 2 + in_k]
        out_k_str = utils.LETTERS[2 + in_k : 2 + in_k + out_k]
        # (in,tensor_in), (out,in,tensor_in_out) -> (out,tensor_out)
        out = jnp.einsum(f"a{in_k_str},ba{in_k_str}{out_k_str}->b{out_k_str}", x, tensor_map)

        if self.use_bias:
            # (out,len_bias), (len_bias,tensor_out)
            out = out + jnp.einsum(
                "ab,b...->a...", self.bias, jax.lax.stop_gradient(self.bias_basis)
            )

        return out

    def count_params(self: Self):
        return self.weights.size + self.bias.size


class PermEquivariantLayer(eqx.Module):
    layers: dict[tuple[int, int], Union[GeneralLinear, eqx.nn.Linear]]
    input_keys: dict[int, int]
    output_keys: dict[int, int]

    def __init__(
        self: Self, input_keys: dict[int, int], output_keys: dict[int, int], n: int, key: ArrayLike
    ):
        """
        args:
            input_keys: a map from tensor order to number of channels for the inputs
            output_keys: a map from tensor order to number of channels for the outputs
        """
        self.input_keys = input_keys
        self.output_keys = output_keys

        self.layers = {}
        for in_k, in_c in input_keys.items():
            for out_k, out_c in output_keys.items():
                key, subkey = random.split(key)
                if in_k + out_k == 0:
                    self.layers[(in_k, out_k)] = eqx.nn.Linear(in_c, out_c, key=subkey)
                else:
                    sym_axes = ()
                    if in_k > 1 and out_k > 1:
                        sym_axes = (tuple(range(in_k)), tuple(range(in_k, in_k + out_k)))
                    elif in_k > 1:
                        sym_axes = (tuple(range(in_k)),)
                    elif out_k > 1:
                        sym_axes = (tuple(range(out_k)),)

                    # this class saves the perm invariant tensors, so they aren't reconstructed
                    basis = utils.PermInvariantTensor.get(in_k + out_k, n, sym_axes)
                    if out_k > 0:
                        bias_basis = utils.PermInvariantTensor.get(
                            out_k, n, (tuple(range(out_k)),) if out_k > 1 else ()
                        )
                    else:
                        bias_basis = jnp.ones((1))

                    self.layers[(in_k, out_k)] = GeneralLinear(
                        basis, in_c, out_c, True, bias_basis, subkey
                    )

    def __call__(self: Self, x: dict[int, ArrayLike]):
        out_dict = {}
        for in_k, tensor in x.items():
            for out_k in self.output_keys.keys():
                if out_k in out_dict:
                    out_dict[out_k] = out_dict[out_k] + self.layers[(in_k, out_k)](tensor)
                else:
                    out_dict[out_k] = self.layers[(in_k, out_k)](tensor)

        return out_dict

    def count_params(self: Self):
        return sum(
            [
                (
                    layer.count_params()
                    if isinstance(layer, GeneralLinear)
                    else sum(
                        [x.size for x in jax.tree_util.tree_leaves(eqx.filter(layer, eqx.is_array))]
                    )
                )
                for layer in self.layers.values()
            ]
        )


## Data and Batching operations


def get_batches(X, y, batch_size, rand_key, devices):
    devices = jax.devices() if devices is None else devices

    X_batches = []
    y_batches = []
    batch_indices = random.permutation(rand_key, len(X))
    # if total size is not divisible by batch, the remainder will be ignored
    for i in range(int(math.floor(len(X) / batch_size))):  # iterate through the batches of an epoch
        idxs = batch_indices[i * batch_size : (i + 1) * batch_size]
        # shape to (num_gpus,batch/num_gpus,...)
        X_batches.append(X[idxs].reshape((len(devices), -1) + X[idxs].shape[1:]))
        y_batches.append(y[idxs].reshape((len(devices), -1) + y[idxs].shape[1:]))

    return X_batches, y_batches


## Training


class StopCondition:
    def __init__(self, verbose=0) -> None:
        assert verbose in {0, 1, 2}
        self.best_model = None
        self.verbose = verbose

    def stop(self, model, current_epoch, train_loss, val_loss, epoch_time):
        pass

    def log_status(self, epoch, train_loss, val_loss, epoch_time, emph=False):
        if train_loss is not None:
            if val_loss is not None:
                print(
                    f"{'> ' if emph else ''}Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Epoch time: {epoch_time:.5f}",
                )
            else:
                print(f"Epoch {epoch} Train: {train_loss:.7f} Epoch time: {epoch_time:.5f}")


class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self, epochs, verbose=0) -> None:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(self, model, current_epoch, train_loss, val_loss, batch_time) -> bool:
        self.best_model = model

        if self.verbose == 2 or (
            self.verbose == 1 and (current_epoch % (self.epochs // min(10, self.epochs)) == 0)
        ):
            self.log_status(current_epoch, train_loss, val_loss, batch_time)

        return current_epoch >= self.epochs


class TrainLoss(StopCondition):
    # Stop when the training error stops improving after patience number of epochs.

    def __init__(self, patience=0, min_delta=0, verbose=0) -> None:
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(self, model, current_epoch, train_loss, val_loss, batch_time) -> bool:
        if train_loss is None:
            return False

        if train_loss < (self.best_train_loss - self.min_delta):
            self.best_train_loss = train_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, batch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience


class ValLoss(StopCondition):
    # Stop when the validation error stops improving after patience number of epochs.

    def __init__(self, patience=0, min_delta=0, verbose=0) -> None:
        super(ValLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(self, model, current_epoch, train_loss, val_loss, batch_time) -> bool:
        if val_loss is None:
            return False

        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, batch_time, emph=True)
        else:
            self.epochs_since_best += 1
            if self.verbose >= 2:
                self.log_status(current_epoch, train_loss, val_loss, batch_time)

        return self.epochs_since_best > self.patience


def save(filename, model):
    # TODO: save batch stats
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def evaluate(
    model: eqx.Module,
    map_and_loss: Union[
        Callable[
            [eqx.Module, ArrayLike, ArrayLike, eqx.nn.State],
            tuple[ArrayLike, eqx.nn.State, ArrayLike],
        ],
        Callable[
            [eqx.Module, ArrayLike, ArrayLike, eqx.nn.State],
            tuple[ArrayLike, eqx.nn.State],
        ],
    ],
    x: ArrayLike,
    y: ArrayLike,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> jax.Array:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the data is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        model (model PyTree): the model to run through map_and_loss
        x (array): input data
        y (array): target output data
        sharding: sharding over multiple GPUs, if None (default), will use available devices
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
    returns: average loss over the entire layer
    """
    inference_model = eqx.nn.inference_mode(model)
    if return_map:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None, 0),
        )
        loss, _, out = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0), out.reshape(-1)
    else:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None),
        )
        loss, _ = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0)


def loss_reducer(ls):
    """
    A reducer for map_loss_in_batches that takes the batch mean of the loss
    """
    return jnp.mean(jnp.stack(ls), axis=0)


def aux_data_reducer(ls):
    """
    A reducer for aux_data like batch stats that just takes the last one
    """
    return ls[-1]


def data_reducer(ls):
    """
    If map data returns the mapped data, merge them togther
    """
    # TODO: fix this
    return functools.reduce(lambda carry, val: carry.concat(val), ls, ls[0].empty())


def map_loss_in_batches(
    map_and_loss: Callable[
        [eqx.Module, ArrayLike, ArrayLike, eqx.nn.State], tuple[jax.Array, eqx.nn.State]
    ],
    model: eqx.Module,
    x: ArrayLike,
    y: ArrayLike,
    batch_size: int,
    rand_key: ArrayLike,
    reducers: Optional[tuple] = None,
    devices: Optional[list[jax.devices]] = None,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> jax.Array:
    """
    Runs map_and_loss for the entire x,y, splitting into batches if the data is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as any remainder of the data.
    args:
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model (model PyTree): the model to run through map_and_loss
        x (array): input data
        y (array): target output data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        devices (list of jax devices): the gpus that the code will run on
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
    returns: average loss over the entire dataset
    """
    if reducers is None:
        # use the default reducer for loss
        reducers = [loss_reducer]
        if return_map:
            reducers.append(data_reducer)

    X_batches, Y_batches = get_batches(x, y, batch_size, rand_key, devices)
    results = [[] for _ in range(len(reducers))]
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        one_result = evaluate(model, map_and_loss, X_batch, Y_batch, aux_data, return_map)

        if len(reducers) == 1:
            results[0].append(one_result)
        else:
            for val, result_ls in zip(one_result, results):
                result_ls.append(val)

    if len(reducers) == 1:
        return reducers[0](results[0])
    else:
        return tuple(reducer(result_ls) for reducer, result_ls in zip(reducers, results))


def train_step(
    map_and_loss: Callable[
        [eqx.Module, ArrayLike, ArrayLike, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state,
    x: ArrayLike,
    y: ArrayLike,
    aux_data: Optional[eqx.nn.State] = None,
):
    """
    Perform one step and gradient update of the model. Uses filter_pmap to use multiple gpus.
    args:
        map_and_loss (func): map and loss function where the input is a model pytree, x BatchLayer,
            y BatchLayer, and aux_data, and returns a float loss and aux_data
        model (equinox model pytree): the model
        optim (optax optimizer):
        opt_state:
        x (array): input data
        y (array): target data
        aux_data (Any): auxilliary data for stateful layers
    returns: model, opt_state, loss_value
    """
    # NOTE: do not `jit` over `pmap` see (https://github.com/google/jax/issues/2926)
    loss_grad = eqx.filter_value_and_grad(map_and_loss, has_aux=True)

    compute_loss_pmap = eqx.filter_pmap(
        loss_grad,
        axis_name="pmap_batch",
        in_axes=(None, 0, 0, None),
        out_axes=((0, None), 0),
    )
    (loss, aux_data), grads = compute_loss_pmap(model, x, y, aux_data)

    loss = jnp.mean(loss, axis=0)

    get_weights = lambda m: jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array)
    new_grad_arrays = [jnp.mean(x, axis=0) for x in get_weights(grads)]
    grads = eqx.tree_at(get_weights, grads, new_grad_arrays)

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss, aux_data


def train(
    X: ArrayLike,
    Y: ArrayLike,
    map_and_loss: Union[
        Callable[[eqx.Module, ArrayLike, ArrayLike], jax.Array],
        Callable[
            [eqx.Module, ArrayLike, ArrayLike, Any],
            tuple[jax.Array, Any],
        ],
    ],
    model: eqx.Module,
    rand_key: ArrayLike,
    stop_condition: StopCondition,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    validation_X: Optional[ArrayLike] = None,
    validation_Y: Optional[ArrayLike] = None,
    save_model: Optional[str] = None,
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> Union[tuple[eqx.Module, Any, jax.Array, jax.Array], tuple[eqx.Module, jax.Array, jax.Array]]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    shards over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    args:
        X (array): The X input data
        Y (array): The Y target data
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        rand_key (jnp.random key): key for randomness
        stop_condition (StopCondition): when to stop the training process, currently only 1 condition
            at a time
        batch_size (int): the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer
        validation_X (array): input data for a validation data set as a layer by k
            of (images, channels, (N,)*D, (D,)*k)
        validation_Y (array): target data for a validation data set as a layer by k
            of (images, channels, (N,)*D, (D,)*k)
        save_model (str): if string, save model every 10 epochs, defaults to None
        aux_data (eqx.nn.State): initial aux data passed in to map_and_loss when has_aux is true.
        devices (list): gpu/cpu devices to use, if None (default) then it will use jax.devices()
    returns: A tuple of best model in inference mode, epoch loss, and val loss
    """
    if isinstance(stop_condition, ValLoss) and (validation_X is None or validation_Y is None):
        raise ValueError("Stop condition is ValLoss, but no validation data provided.")

    devices = devices if devices else jax.devices()

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    val_loss = 0
    epoch_time = 0
    while not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, epoch_time):
        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batches(X, Y, batch_size, subkey, devices)
        epoch_loss = 0
        start_time = time.time()
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            model, opt_state, loss_value, aux_data = train_step(
                map_and_loss,
                model,
                optimizer,
                opt_state,
                X_batch,
                Y_batch,
                aux_data,
            )
            epoch_loss += loss_value

        epoch_loss = epoch_loss / len(X_batches)
        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if validation_X is not None and validation_Y is not None:
            epoch_val_loss = map_loss_in_batches(
                map_and_loss,
                model,
                validation_X,
                validation_Y,
                batch_size,
                subkey,
                devices=devices,
                aux_data=aux_data,
            )
            val_loss = epoch_val_loss

        if save_model and ((epoch % 10) == 0):
            save(save_model, stop_condition.best_model)

        epoch_time = time.time() - start_time

    return stop_condition.best_model, aux_data, epoch_loss, val_loss
