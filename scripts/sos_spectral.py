import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax

import tensorpolynomials.data as tpoly_data
import tensorpolynomials.ml as ml

def sos_method(S):
    """
    The sum-of-squares method from: https://arxiv.org/pdf/1512.02337.pdf
    args:
        S (jnp.array): an (n,d) array, where n is the ambient dimension, 
        d is number of vectors
    """
    n,d = S.shape

    vmap_inner = jax.vmap(lambda ai: (jnp.linalg.norm(ai)**2 - d/n)*jnp.tensordot(ai,ai,axes=0))
    A = jnp.sum(vmap_inner(S),axis=0)
    assert A.shape == (d,d)

    _, eigvecs = jnp.linalg.eigh(A) # ascending order
    assert eigvecs.shape == (d,d)
    u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
    return S @ u

def map_and_loss(model, x, y):
    """
    Map x using the model,
    args:
        x (jnp.array): input data, shape (batch,n,d)
        y (jnp.array): output data, the sparse vector, shape (batch,n)
    """
    pred_y = jax.vmap(model)(x)
    squared_dots = jnp.einsum('...i,...i->...', y, pred_y)**2
    return 1 - jnp.mean(squared_dots)

class SparseVectorHunterDiagonal(eqx.Module):
    layers: list

    def __init__(self, n, width, num_hidden_layers, key):
        """
        Constructor for SparseVectorHunter
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the 
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
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvector
        return S @ u
  
class SparseVectorHunter(eqx.Module):
    layers: list

    def __init__(self, n, width, num_hidden_layers, key):
        """
        Constructor for SparseVectorHunter
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the 
        """
        in_features = n**2
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
        X = (S @ S.T).reshape(-1) # n**2
        assert X.shape == (n**2,)

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

# define data params
n = 10
d = 3
eps = 0.3
train_size = 5000
val_size = 500
test_size = 500

# Random covariance matrix, force it to be pos. def.
# M = random.normal(subkey, shape=(n,n))
# cov = M.T @ M + (1e-5)*jnp.eye(n)

# Random diagonal covariance matrix
cov = jnp.diag(random.uniform(subkey, shape=(n,)))

# No covariance, use 1/n * Id
# cov = None

# define training params
batch_size = 100
learning_rate = 3e-4
trials = 3
verbose = 0

key, subkey1, subkey2 = random.split(key, 3)
models = [
    ('sparseDiagonal', SparseVectorHunterDiagonal(n, 128, 2, subkey1)), # n inputs, n+1 outputs
    ('sparse', SparseVectorHunter(n, 128, 2, subkey2)), # n**2 inputs, n+1+(n(n-1)/2) outputs
]
for model_name, model in models:
    print(f'{model_name}: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,} params')
    
results = np.zeros((trials,len(models)+1,2))

for t in range(trials):
    key, subkey1, subkey2, subkey3 = random.split(key, 4)

    # (batch,n), (batch,n,d)
    train_v0, train_W = tpoly_data.get_synthetic_data(subkey1, n, d, eps, train_size, cov) 
    val_v0, val_W = tpoly_data.get_synthetic_data(subkey2, n, d, eps, val_size, cov)
    test_v0, test_W = tpoly_data.get_synthetic_data(subkey3, n, d, eps, test_size, cov)

    results[t,0,0] = 1 - map_and_loss(sos_method, train_W, train_v0)
    results[t,0,1] = 1 - map_and_loss(sos_method, test_W, test_v0)

    for i, (model_name, model) in enumerate(models):
        print(f'trial: {t}, model: {model_name}')
        key, subkey = random.split(key)
        trained_model, _, _ = ml.train(
            model, 
            map_and_loss, 
            train_W, 
            train_v0, 
            subkey, 
            ml.ValLoss(patience=20, verbose=verbose),
            optax.adam(optax.exponential_decay(learning_rate, int(train_size/batch_size), 0.999)), 
            batch_size, 
            val_W, 
            val_v0,
        )

        results[t,i+1,0] = 1 - map_and_loss(trained_model, train_W, train_v0)
        results[t,i+1,1] = 1 - map_and_loss(trained_model, test_W, test_v0)

print(results)
print(jnp.mean(results, axis=0))
