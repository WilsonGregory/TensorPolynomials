import time
import math

import jax.numpy as jnp
import jax.random as random
import equinox as eqx

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

    def log_status(self, epoch, train_loss, val_loss, batch_time):
        if (train_loss is not None):
            if (val_loss is not None):
                print(
                    f'Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Batch time: {batch_time:.5f}',
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
            (self.verbose == 1 and (current_epoch % (self.epochs // jnp.min([10,self.epochs])) == 0))
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
                self.log_status(current_epoch, train_loss, val_loss, batch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience

def make_step(map_and_loss, model, optim, opt_state, x, y):
    loss_value, grads = eqx.filter_value_and_grad(map_and_loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

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
    opt_state = optim.init(model)

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

        if (validation_X is not None) and (validation_Y is not None):
            epoch_val_loss = map_and_loss(model, validation_X, validation_Y)
            val_loss.append(epoch_val_loss)

    return stop_condition.best_model, jnp.array(train_loss), jnp.array(val_loss)
