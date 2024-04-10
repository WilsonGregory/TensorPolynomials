from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as random

def get_sparse_vector(key, n, eps, batch):
    """
    This function uses the accept/reject method to find an L4 sparse vector. First we generate an
    L2 unit vector in R^n. Then we accept it if norm(v)_4^4 >= 1/eps*n
    args:
        key (rand key):
        n (int): vector dimension
        eps (float): sparsity parameter, can think of this as the fraction of components that are
            nonzero, but its actually an L4 relaxation of that concept.
        batch (int): the number of sparse vectors to find
    """
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

    return jnp.stack(vecs)
        
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

def get_synthetic_data(key, n, d, eps, batch, cov=None):
    cov = (1/n)*jnp.eye(n) if cov is None else cov
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    v0 = get_sparse_vector(subkey1, n, eps, batch) # (batch,n)

    V_noise = get_gaussian_vectors(subkey2, n, d-1, batch, cov) # (batch,n,d-1)

    V = jnp.concatenate([v0[...,None], V_noise], axis=-1)
    assert V.shape == (batch, n, d)

    W = get_orthogonal_basis(subkey3, V) # (batch,n,d)

    return v0, W
    