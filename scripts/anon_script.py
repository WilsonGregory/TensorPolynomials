"""
Run this script to reproduce the results from our paper. Since all the necessary code is in this file,
the pip packages you will need are the imports below, namely tqdm, numpy, jax, equinox, and optax.

Modify the arguments around line 600 as necessary if you want to save the results. But they should
work as specified out of the box. On an RTX 6000 ADA GPU, this entire script will take 18 hours. To
do it faster, you can reduce the number of trials from 5 to 1, you can reduce the models or the data
regimes as you please. Note that the GPU we used has 48 GB of memory, so if you have a smaller GPU
you may also have to reduce the number of training data points because we are lazy and just store them
all on the GPU.

You can change the verbose level from 0 to 1 to see the validation error as the models train.
"""

import time
import math

from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax

TINY = 1e-5

V0_NORMAL = 'Accept/Reject'
V0_BERN_GAUS = 'Bernoulli-Gaussian'
V0_BERN_DUB_GAUS = 'Bernoulli-Dual-Gaussian'
V0_BERN_RAD = 'Bernoulli-Rademacher'
V0_KSPARSE = 'v0_ksparse'

## Part 1: Data Generation Code
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def nonzero_and_norm(vecs):
    """
    Take vecs, remove those with 0 norm, then normalize the rest.
    args:
        vecs (jnp.array): vecs of shape (batch,n)
    """
    norms = jnp.linalg.norm(vecs, axis=1) # (batch,)
    vecs = vecs[norms > TINY]
    return vecs / norms[norms > TINY,None]

def get_sparse_vector(key, n, eps, batch, v0_sampling=V0_NORMAL):
    """
    Generate batch sparse vectors in R^n according to different strategies.
    V0_NORMAL: generate a unit-length gaussian vector, accept if norm(v)_4^4 >= 1/eps*n else reject
    V0_BERN_GAUS: with prob eps, v0_i is ~N(0,1/n*eps), else 0
    V0_BERN_RAD: with prob eps/2, v0_i is 1/sqrt(n*eps), prob eps/2 v0_i is -1/sqrt(n*eps), else 0
    V0_KSPARSE: select eps*n entries to be nonzero, sample those from a gaussian
    No matter the sampling, we always set the vectors to unit length after generating.
    args:
        key (rand key):
        n (int): vector dimension
        eps (float): sparsity parameter, can think of this as the fraction of components that are
            nonzero, but its actually an L4 relaxation of that concept.
        batch (int): the number of sparse vectors to find
        v0_sampling (string): one of V0_NORMAL, V0_BERN_GAUS, V0_BERN_DUB_GAUS, or V0_BERN_RAD
    """
    if v0_sampling == V0_NORMAL:
        print(f'generating {batch} sparse vectors...')
        vecs = []
        with tqdm(total=batch) as pbar:
            while len(vecs) < batch:
                key, subkey = random.split(key)
                vec = random.normal(subkey, shape=(n,))
                vec = vec/jnp.linalg.norm(vec)

                if (jnp.sum(vec**4) >= (1/(eps*n))):
                    vecs.append(vec)
                    pbar.update(1)

        sparse_vecs = jnp.stack(vecs)

    elif v0_sampling == V0_BERN_GAUS:
        sparse_vecs = jnp.zeros((0,n))

        while len(sparse_vecs) < batch:
            key, subkey1, subkey2 = random.split(key, 3)
            entries = random.multivariate_normal(
                subkey1, 
                jnp.zeros(n), 
                (1/(n*eps))*jnp.eye(n), 
                shape=(batch,),
            )
            bernoulli = random.uniform(subkey2, shape=(batch,n))
            vecs = jnp.where(bernoulli < eps, entries, jnp.zeros(entries.shape))

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_BERN_DUB_GAUS:
        sparse_vecs = jnp.zeros((0,n))
        assert eps <= (1/3)
        q = jnp.sqrt((1/3)*(1-eps)*(1-3*eps))

        while len(sparse_vecs) < batch:
            key, subkey1, subkey2, subkey3 = random.split(key, 4)
            big_entries = random.multivariate_normal(
                subkey1, 
                jnp.zeros(n), 
                ((eps+q)/(n*eps))*jnp.eye(n), 
                shape=(batch,),
            )
            little_entries = random.multivariate_normal(
                subkey2,
                jnp.zeros(n),
                ((1 - eps - q)/(n*(1-eps)))*jnp.eye(n),
                shape=(batch,),
            )  

            bernoulli = random.uniform(subkey3, shape=(batch,n))
            vecs = jnp.where(bernoulli < eps, big_entries, little_entries)

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_BERN_RAD:
        sparse_vecs = jnp.zeros((0,n))

        while len(sparse_vecs) < batch:
            key, subkey = random.split(key)
            bernoulli = random.uniform(subkey, shape=(batch,n))
            entries = jnp.ones((batch,n))/jnp.sqrt(n*eps)
            signed_entries = jnp.where(bernoulli < 0.5, entries, -1*entries)
            vecs = jnp.where(bernoulli < eps, signed_entries, jnp.zeros(signed_entries.shape))

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_KSPARSE:
        key, subkey1, subkey2 = random.split(key, 3)
        num_nonzero = np.floor(eps*n).astype(int) # exact number of nonzero entries per vector
        
        random.normal(subkey1, shape=((batch,num_nonzero)))
        entries = jnp.concatenate(
            [random.normal(subkey1, shape=(batch,num_nonzero)), jnp.zeros((batch,n - num_nonzero))],
            axis=1,
        )
        shuffled_entries = random.permutation(subkey2, entries, axis=1, independent=True)
        sparse_vecs = shuffled_entries / jnp.linalg.norm(shuffled_entries, axis=1, keepdims=True)

    return sparse_vecs
        
def get_gaussian_vectors(key, n, num_vecs, batch, cov):
    V = random.multivariate_normal(key, jnp.zeros(n), cov, shape=(batch,num_vecs)).transpose((0,2,1))
    assert V.shape == (batch,n,num_vecs)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True) # (batch,1,num_vecs)
    assert norms.shape == (batch,1,num_vecs)
    return V / norms

def get_orthogonal_basis(key, V):
    """
    args:
        key (rand key): we want a random orthogonal basis
        V (jnp.array): shape (batch,n,d)
    """
    batch,n,d = V.shape
    O = random.orthogonal(key, d, shape=(batch,))
    assert O.shape == (batch,d,d)
    Q,_ = jnp.linalg.qr(jnp.einsum('...ij,...jk->...ik', V, O))
    assert Q.shape == (batch,n,d)
    return Q

def get_synthetic_data(key, n, d, eps, batch, v0_sampling=V0_NORMAL, cov=None):
    isnan = True
    while (isnan):
        cov = (1/n)*jnp.eye(n) if cov is None else cov
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        v0 = get_sparse_vector(subkey1, n, eps, batch, v0_sampling) # (batch,n)

        V_noise = get_gaussian_vectors(subkey2, n, d-1, batch, cov) # (batch,n,d-1)

        V = jnp.concatenate([v0[...,None], V_noise], axis=-1)
        assert V.shape == (batch, n, d)

        W = get_orthogonal_basis(subkey3, V) # (batch,n,d)

        # Sometimes this is nan, I am not sure why. It also breaks the other models when
        # this happens. I thought it might be some issue with normalizing/dividing by zero
        # but it keeps happening. So for now, we will just do this.
        isnan = jnp.isnan(1 - map_and_loss(sos_method, W, v0))

    return v0, W


## Part 2: Machine Learning Code
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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



## Part 3: Models and Script
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RAND_COV = 'Random'
DIAG_COV = 'Diagonal'
UNIT_COV = 'Identity'

sampling_print_names = {
    V0_NORMAL: 'A/R',
    V0_BERN_GAUS: 'BG',
    V0_BERN_DUB_GAUS: 'CBG',
    V0_BERN_RAD: 'BR',
    V0_KSPARSE: 'KS',
}

def sos_method(S):
    """
    The sum-of-squares method from: https://arxiv.org/pdf/1512.02337.pdf
    args:
        S (jnp.array): an (n,d) array, where n is the ambient dimension, d is number of vectors
    """
    n,d = S.shape

    vmap_inner = jax.vmap(lambda ai: (jnp.linalg.norm(ai)**2 - d/n)*jnp.tensordot(ai,ai,axes=0))
    A = jnp.sum(vmap_inner(S),axis=0)
    assert A.shape == (d,d)

    _, eigvecs = jnp.linalg.eigh(A) # ascending order
    assert eigvecs.shape == (d,d)
    u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
    return S @ u

def sos_methodII(S):
    """
    The sum-of-squares method from the newer: https://arxiv.org/pdf/2105.15081.pdf
    We match the notation of that paper with (N,n) rather than (n,d)
    args:
        S (jnp.array): an (N,n) array, where N is the ambient dimension and n is num of vectors
    """
    N,n = S.shape

    vmap_inner = jax.vmap(lambda ai: (jnp.linalg.norm(ai)**2 - ((n-1)/N))*jnp.tensordot(ai,ai,axes=0))
    A = jnp.sum(vmap_inner(S),axis=0) - 3*jnp.eye(n)
    assert A.shape == (n,n)

    _, eigvecs = jnp.linalg.eigh(A) # ascending order
    assert eigvecs.shape == (n,n)
    u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
    return S @ u

def map_and_loss(model, x, y):
    """
    Map x using the model,
    args:
        model (functional): function on a single input, will be vmapped
        x (jnp.array): input data, shape (batch,n,d)
        y (jnp.array): output data, the sparse vector, shape (batch,n)
    """
    pred_y = jax.vmap(model)(x)
    squared_dots = jnp.einsum('...i,...i->...', y, pred_y)**2
    return 1 - jnp.mean(squared_dots)

def fill_triangular(x, d):
    """
    Constructs an upper triangular matrix from x: https://github.com/google/jax/discussions/10146
    Note that the order is a little funky, for example a 5x5 from 15 values will be:
    array([[ 1,  2,  3,  4,  5],
            [ 0,  7,  8,  9, 10],
            [ 0,  0, 13, 14, 15],
            [ 0,  0,  0, 12, 11],
            [ 0,  0,  0,  0,  6]])
    args:
        x (jnp.array): array of values that we want to become the upper triangular matrix
        d (int): the dimensions of the matrix we are making, d x d
    """
    assert len(x) == (d + d*(d-1)//2)
    xc = jnp.concatenate([x, x[d:][::-1]])
    y = jnp.reshape(xc, [d, d])
    return jnp.triu(y, k=0)

class BaselineLearnedModel(eqx.Module):
    layers: list

    def __init__(self, n, d, width, num_hidden_layers, key):
        """
        Constructor for SparseVectorHunter that only works with inner product of rows with itself
        as inputs, and a basis of each rows outer product with itself.
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the random key
        """
        in_features = n * d
        out_features = d + (d * (d-1) // 2) # output array needs to be symmetric

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Linear(in_features, width, key=subkey)]
        for _ in range(num_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(width, width, key=subkey))

        key, subkey = random.split(key)
        self.layers.append(eqx.nn.Linear(width, out_features, key=subkey))

    def __call__(self, S):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        n,d = S.shape
        X = S.reshape(-1)
        assert X.shape == (n*d,)

        for layer in self.layers[:-1]:
            X = jax.nn.relu(layer(X))

        out = self.layers[-1](X)
        upper_triangular = fill_triangular(out, d)
        # subtract the diagonal because it will get added twice otherwise
        A = upper_triangular + upper_triangular.T - jnp.diag(upper_triangular)
        assert A.shape == (d,d)

        _, eigvecs = jnp.linalg.eigh(A) # ascending order
        assert eigvecs.shape == (d,d)
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
        return S @ u

class SparseVectorHunterDiagonal(eqx.Module):
    layers: list

    def __init__(self, n, width, num_hidden_layers, key):
        """
        Constructor for SparseVectorHunter that only works with inner product of rows with itself
        as inputs, and a basis of each rows outer product with itself.
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the random key
        """
        in_features = n
        out_features = n + 1

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Linear(in_features, width, key=subkey)]
        for _ in range(num_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(width, width, key=subkey))

        key, subkey = random.split(key)
        self.layers.append(eqx.nn.Linear(width, out_features, key=subkey))

    def __call__(self, S):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        n,d = S.shape
        X = jnp.diag(S @ S.T).reshape(-1)
        assert X.shape == (n,)

        for layer in self.layers[:-1]:
            X = jax.nn.relu(layer(X))

        out = self.layers[-1](X)

        # (n,d) -> (n,d,d)
        outer_prods = jax.vmap(lambda row: jnp.tensordot(row,row,axes=0))(S)
        basis = jnp.concatenate([outer_prods, jnp.eye(d)[None]])
        assert basis.shape == (n+1,d,d)
        A = jnp.sum(out.reshape((-1,1,1))*basis, axis=0) 

        _, eigvecs = jnp.linalg.eigh(A) # ascending order
        assert eigvecs.shape == (d,d)
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
        return S @ u

class SparseVectorHunter(eqx.Module):
    layers: list

    def __init__(self, n, width, num_hidden_layers, key):
        """
        Constructor for SparseVectorHunter, parameterizes full function from vectors to 2-tensor.
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the  random key
        """
        in_features = n + (n*(n-1)//2)
        out_features = n + 1 + (n*(n-1)//2)

        key, subkey = random.split(key)
        self.layers = [eqx.nn.Linear(in_features, width, key=subkey)]
        for _ in range(num_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(eqx.nn.Linear(width, width, key=subkey))

        key, subkey = random.split(key)
        self.layers.append(eqx.nn.Linear(width, out_features, key=subkey))

    def __call__(self, S):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        n,d = S.shape
        X = (S @ S.T)[jnp.triu_indices(n)].reshape(-1)
        assert X.shape == (n + n*(n-1)//2,)

        for layer in self.layers[:-1]:
            X = jax.nn.relu(layer(X))

        out = self.layers[-1](X)

        # Now get a basis consisting of outer products of all pairs of rows. Combine the
        # a_i a_j with a_j a_i so that the basis is Hermitian

        # (d,n,n,d) -> (d,d,n,n) -> (d**2,n,n)
        outer_prods = jnp.tensordot(S.T,S,axes=0).transpose((0,3,1,2)).reshape((d**2,n,n))
        outer_prods = jax.vmap(lambda arr: arr[jnp.triu_indices(n)])(outer_prods).reshape((d,d,-1))
        hermitian_prods = jnp.moveaxis(0.5*(outer_prods + outer_prods.transpose((1,0,2))), 2, 0)

        basis = jnp.concatenate([hermitian_prods, jnp.eye(d)[None]])
        assert basis.shape == (n + 1 + (n*(n-1)//2),d,d)

        A = jnp.sum(out.reshape((-1,1,1))*basis, axis=0)

        _, eigvecs = jnp.linalg.eigh(A) # ascending order
        assert eigvecs.shape == (d,d)
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvector
        return S @ u

# Main
key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
train_model = True # set to false if it is already trained and you want to load the data
save = False # set to true to save the data
save_dir = '../runs/' # location of the saved data
table_print = False # leave this one False unless you are copying the results into a latex table.

# define data params
n = 100
d = 5
eps = 0.25
train_size = 5000
val_size = 500
test_size = 500

# define training params
width = 128
depth = 2
batch_size = 100
trials = 5
verbose = 0 # change to 1 to see the validation error as we train

key, subkey1, subkey2, subkey3 = random.split(key, 4)
models = [
    ('baselineLearned', 1e-3, BaselineLearnedModel(n, d, width, depth, subkey1)), # n*d inputs, d*d outputs
    ('sparseDiagonal', 5e-4, SparseVectorHunterDiagonal(n, width, depth, subkey2)), # n inputs, n+1 outputs
    ('sparse', 3e-4, SparseVectorHunter(n, width, depth, subkey3)), # n+(n(n-1)/2) inputs, n+1+(n(n-1)/2) outputs
]
for model_name, lr, model in models:
    print(f'{model_name}: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,} params')

samplings = [
    V0_NORMAL,
    V0_BERN_GAUS,
    V0_BERN_DUB_GAUS,
    V0_BERN_RAD,
]
covariances = [RAND_COV, DIAG_COV, UNIT_COV]

results = np.zeros((trials,len(samplings),len(covariances),len(models)+2,2))

if train_model:
    for t in range(trials):
        for i, v0_sampling in enumerate(samplings):
            for j, cov_type in enumerate(covariances):
                key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

                if cov_type == RAND_COV:
                    # Random covariance matrix, force it to be pos. def.
                    M = random.normal(subkey4, shape=(n,n))
                    cov = M.T @ M + (1e-5)*jnp.eye(n)
                elif cov_type == DIAG_COV:
                    # Random diagonal covariance matrix
                    cov = jnp.diag(random.uniform(subkey, shape=(n,), minval=0.5, maxval=1.5))
                elif cov_type == UNIT_COV:
                    # No covariance, use 1/n * Id
                    cov = None
                else:
                    raise Exception(f'No covariance {cov}')

                # (batch,n), (batch,n,d)
                train_v0, train_W = get_synthetic_data(subkey1, n, d, eps, train_size, v0_sampling, cov) 
                val_v0, val_W = get_synthetic_data(subkey2, n, d, eps, val_size, v0_sampling, cov)
                test_v0, test_W = get_synthetic_data(subkey3, n, d, eps, test_size, v0_sampling, cov)

                results[t,i,j,0,0] = 1 - map_and_loss(sos_method, train_W, train_v0)
                results[t,i,j,0,1] = 1 - map_and_loss(sos_method, test_W, test_v0)
                print(f'{t},{v0_sampling},{cov_type},sos: {results[t,i,j,0,1]}')

                results[t,i,j,1,0] = 1 - map_and_loss(sos_methodII, train_W, train_v0)
                results[t,i,j,1,1] = 1 - map_and_loss(sos_methodII, test_W, test_v0)
                print(f'{t},{v0_sampling},{cov_type},sosII: {results[t,i,j,1,1]}')

                for k, (model_name, lr, model) in enumerate(models):
                    key, subkey = random.split(key)
                    trained_model, _, _ = train(
                        model, 
                        map_and_loss, 
                        train_W, 
                        train_v0, 
                        subkey, 
                        ValLoss(patience=20, verbose=verbose),
                        optax.adam(optax.exponential_decay(lr, int(train_size/batch_size), 0.999)), 
                        batch_size, 
                        val_W, 
                        val_v0,
                    )

                    results[t,i,j,k+2,0] = 1 - map_and_loss(trained_model, train_W, train_v0)
                    results[t,i,j,k+2,1] = 1 - map_and_loss(trained_model, test_W, test_v0)
                    print(f'{t},{v0_sampling},{cov_type},{model_name}: {results[t,i,j,k+2,1]}')
                
                if save:
                    jnp.save(
                        f'{save_dir}/sparse_vector_results_t{trials}_N{train_size}_n{n}_d{d}_eps{eps}.npy',
                        results,
                    )
else:
    results = jnp.load(f'{save_dir}/sparse_vector_results_t{trials}_N{train_size}_n{n}_d{d}_eps{eps}.npy')

mean_results = jnp.mean(results, axis=0)
std_results = jnp.std(results, axis=0)
print_models = [('sos', sos_method), ('sosII', sos_methodII)] + models
if table_print:
    mean_results = jnp.around(mean_results, 3)
    std_results = jnp.around(std_results, 3)

    for l in [0,1]: # train is 0, test is 1
        if l == 0:
            print('Train')
        else:
            print('Test')

        print('\\hline')
        for i, v0_sampling in enumerate(samplings):
            for j, cov_type in enumerate(covariances):
                if j == (len(covariances) // 2):
                    print(f'{sampling_print_names[v0_sampling]} ', end='')

                print(f'& {cov_type} ', end='')

                for k in range(len(print_models)):
                    if jnp.allclose(mean_results[i,j,k,l], jnp.max(mean_results[i,j,:,l])):
                        print(f'& \\textbf{"{"}{mean_results[i,j,k,l]:.3f} $\\pm$ {std_results[i,j,k,l]:.3f}{"}"} ', end='')
                    else:
                        print(f'& {mean_results[i,j,k,l]:.3f} $\\pm$ {std_results[i,j,k,l]:.3f} ', end='')

                print('\\\\')

            print('\\hline')
        
        print('\n')
else:
    for i, v0_sampling in enumerate(samplings):
        print(f'\nv0_sampling: {v0_sampling}')
        for j, cov_type in enumerate(covariances):
            print(f'cov_type: {cov_type}')
            print(mean_results[i,j])
            print(f'+ {std_results[i,j]}')


