import time
import math
import functools

import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array
import equinox as eqx

import tensorpolynomials.utils as utils

class GeneralLinear(eqx.Module):
    """
    Performs a linear (affine) map, using a specified basis rather than the usual linear basis.
    The basis for both the linear map and the bias must be specified.
    """
    basis: Array = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    bias_basis: Array = eqx.field(static=True)
    weights: Array
    bias: Array

    def __init__(
            self, 
            basis: Array, 
            in_features: int,
            out_features: int, 
            use_bias: bool = True, 
            bias_basis: Array = None,
            key = None,
        ):
        self.basis = basis
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias_basis = bias_basis

        wkey, bkey = random.split(key)

        lim = 1 / math.sqrt(in_features*len(self.basis))
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

    def __call__(self, x: Array, key = None):
        in_k = x.ndim - 1
        out_k = self.basis.ndim - in_k - 1
        tensor_map = jnp.einsum('abc,c...->ab...', self.weights, self.basis) # (out,in,) + (n,)*k

        in_k_str = utils.LETTERS[2:2+in_k]
        out_k_str = utils.LETTERS[2+in_k:2+in_k+out_k]
        out = jnp.einsum(f'a{in_k_str},ba{in_k_str}{out_k_str}->b{out_k_str}', x, tensor_map)

        if self.use_bias:
            out = out + jnp.einsum('ab,b...->a...', self.bias, self.bias_basis)
        
        return out

## Data and Batching operations

def get_batches(X, y, batch_size, rand_key):
    batches = []
    batch_indices = random.permutation(rand_key, len(X))
    # if total size is not divisible by batch, the remainder will be ignored
    for i in range(int(math.floor(len(X) / batch_size))): #iterate through the batches of an epoch
        idxs = batch_indices[i*batch_size:(i+1)*batch_size]

        batches.append((X[idxs], y[idxs]))

    return batches

## Training

class StopCondition:
    def __init__(self, verbose=0) -> None:
        assert verbose in {0, 1, 2}
        self.best_model = None
        self.verbose = verbose

    def stop(self, model, current_epoch, train_loss, val_loss, batch_time):
        pass

    def log_status(self, epoch, train_loss, val_loss, batch_time, emph=False):
        if (train_loss is not None):
            if (val_loss is not None):
                print(
                    f'{'> ' if emph else ''}Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Batch time: {batch_time:.5f}',
                )
            else:
                print(f'Epoch {epoch} Train: {train_loss:.7f} Batch time: {batch_time:.5f}')

class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self, epochs, verbose=0) -> None:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(self, model, current_epoch, train_loss, val_loss, batch_time) -> bool:
        self.best_model = model

        if (
            self.verbose == 2 or
            (self.verbose == 1 and (current_epoch % (self.epochs // min(10,self.epochs)) == 0))
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
        if (train_loss is None):
            return False
        
        if (train_loss < (self.best_train_loss - self.min_delta)):
            self.best_train_loss = train_loss
            self.best_model = model
            self.epochs_since_best = 0

            if (self.verbose >= 1):
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
        if (val_loss is None):
            return False
        
        if (val_loss < (self.best_val_loss - self.min_delta)):
            self.best_val_loss = val_loss
            self.best_model = model
            self.epochs_since_best = 0

            if (self.verbose >= 1):
                self.log_status(current_epoch, train_loss, val_loss, batch_time, emph=True)
        else:
            self.epochs_since_best += 1
            if self.verbose >= 2:
                self.log_status(current_epoch, train_loss, val_loss, batch_time)

        return self.epochs_since_best > self.patience

def make_step(map_and_loss, model, optim, opt_state, x, y):
    loss_value, grads = eqx.filter_value_and_grad(map_and_loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

def map_loss_in_batches(
    map_and_loss,
    model,
    X,
    Y, 
    batch_size,
    rand_key,
):
    """
    Runs map_and_loss for the entire layer_X, layer_Y, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number 
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_and_loss (function): function that takes in params, X_batch, Y_batch, rand_key, train, and 
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        params (params tree): the params to run through map_and_loss
        layer_X (BatchLayer): input data
        layer_Y (BatchLayer): target output data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        train (bool): whether this is training or not, likely not
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        devices (list): gpu/cpu devices to use
    returns: average loss over the entire layer
    """
    batches = get_batches(X, Y, batch_size, rand_key)
    total_loss = functools.reduce(lambda carry, batch: carry + map_and_loss(model, *batch), batches, 0)
    return total_loss / len(batches)

def train(
    model, 
    map_and_loss, 
    X, 
    Y, 
    key, 
    stop_condition, 
    optim, 
    batch_size, 
    validation_X=None, 
    validation_Y=None,
):
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The params are returned along with the train_loss
    and val_loss.
    args:
        model (function): subclass of eqx.nn.Module, has a __call__ operator
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and returns the loss.
        X (jnp.array): The X input data, shape (batch,...)
        Y (jnp.array): The Y target data, shape (batch,...)
        rand_key (jnp.random key): key for randomness
        stop_condition (StopCondition): when to stop the training process, currently only 1 condition
            at a time
        batch_size (int): defaults to 16, the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer, like optax.adam(3e-4)
        validation_X (jnp.array): The X input data for a validation set, shape (batch,...)
        validation_Y (jnp.array): The Y target data for a validation set, shape (batch,...)
        save_params (str): if string, save params every 10 epochs, defaults to None
        devices (list): gpu/cpu devices to use, if None (default) then it will use jax.devices()
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    train_loss = []
    val_loss = []
    batch_times = []
    while (
        not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, jnp.mean(jnp.array(batch_times)))
    ):
        batch_times = []
        epoch_loss = 0
        key, subkey = random.split(key)
        batches = get_batches(X, Y, batch_size, subkey)
        for batch_X, batch_Y in batches:
            start_time = time.time()
            model, opt_state, loss = make_step(map_and_loss, model, optim, opt_state, batch_X, batch_Y)
            batch_times.append(time.time() - start_time)
            epoch_loss += loss

        epoch_loss = epoch_loss / len(batches)
        train_loss.append(epoch_loss)
        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if (validation_X is not None and validation_Y is not None):
            key, subkey = random.split(key)
            epoch_val_loss = map_loss_in_batches(map_and_loss, model, validation_X, validation_Y, batch_size, subkey)
            val_loss.append(epoch_val_loss)

    return stop_condition.best_model, jnp.array(train_loss), jnp.array(val_loss)
