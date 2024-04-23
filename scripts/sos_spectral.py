import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax

import tensorpolynomials.data as tpoly_data
import tensorpolynomials.ml as ml

RAND_COV = 'Random'
DIAG_COV = 'Diagonal'
UNIT_COV = 'Identity'

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
train = False
save = False
save_dir = '../runs/'
table_print = True

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
learning_rate = 3e-4
trials = 10
verbose = 0

key, subkey1, subkey2 = random.split(key, 3)
models = [
    ('sparseDiagonal', SparseVectorHunterDiagonal(n, width, depth, subkey1)), # n inputs, n+1 outputs
    ('sparse', SparseVectorHunter(n, width, depth, subkey2)), # n**2 inputs, n+1+(n(n-1)/2) outputs
]
for model_name, model in models:
    print(f'{model_name}: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,} params')
    
# samplings = [tpoly_data.V0_NORMAL,tpoly_data.V0_BERN_GAUS,tpoly_data.V0_BERN_RAD,tpoly_data.V0_KSPARSE]
samplings = [tpoly_data.V0_NORMAL,tpoly_data.V0_BERN_GAUS,tpoly_data.V0_BERN_RAD]
covariances = [RAND_COV, DIAG_COV, UNIT_COV]

results = np.zeros((trials,len(samplings),len(covariances),len(models)+2,2))

if train:
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
                    cov = jnp.diag(random.uniform(subkey, shape=(n,)))
                elif cov_type == UNIT_COV:
                    # No covariance, use 1/n * Id
                    cov = None
                else:
                    raise Exception(f'No covariance {cov}')

                # (batch,n), (batch,n,d)
                train_v0, train_W = tpoly_data.get_synthetic_data(subkey1, n, d, eps, train_size, v0_sampling, cov) 
                val_v0, val_W = tpoly_data.get_synthetic_data(subkey2, n, d, eps, val_size, v0_sampling, cov)
                test_v0, test_W = tpoly_data.get_synthetic_data(subkey3, n, d, eps, test_size, v0_sampling, cov)

                results[t,i,j,0,0] = 1 - map_and_loss(sos_method, train_W, train_v0)
                results[t,i,j,0,1] = 1 - map_and_loss(sos_method, test_W, test_v0)
                print(f'{t},{v0_sampling},{cov_type},sos: {results[t,i,j,0,1]}')

                results[t,i,j,1,0] = 1 - map_and_loss(sos_methodII, train_W, train_v0)
                results[t,i,j,1,1] = 1 - map_and_loss(sos_methodII, test_W, test_v0)
                print(f'{t},{v0_sampling},{cov_type},sosII: {results[t,i,j,1,1]}')

                for k, (model_name, model) in enumerate(models):
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

                    results[t,i,j,k+2,0] = 1 - map_and_loss(trained_model, train_W, train_v0)
                    results[t,i,j,k+2,1] = 1 - map_and_loss(trained_model, test_W, test_v0)
                    print(f'{t},{v0_sampling},{cov_type},{model_name}: {results[t,i,j,k+2,1]}')
                
                if save:
                    jnp.save(
                        f'{save_dir}/sparse_vector_results_lr{learning_rate}_N{train_size}_n{n}_d{d}_eps{eps}.npy',
                        results,
                    )
else:
    results = jnp.load(f'{save_dir}/sparse_vector_results_lr{learning_rate}_N{train_size}_n{n}_d{d}_eps{eps}.npy')

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
                    print(f'{v0_sampling} ', end='')

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


