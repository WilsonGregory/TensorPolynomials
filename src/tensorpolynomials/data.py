from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

TINY = 1e-5

V0_NORMAL = 'Accept/Reject'
V0_BERN_GAUS = 'Bernoulli-Gaussian'
V0_BERN_RAD = 'Bernoulli-Rademacher'
V0_KSPARSE = 'v0_ksparse'

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
        v0_sampling (string): one of V0_NORMAL, V0_BERN_GAUS, or V0_BERN_RAD
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
            norms = jnp.linalg.norm(vecs, axis=1) # (batch,)

            vecs = vecs[norms > TINY]

            sparse_vecs = jnp.concatenate([sparse_vecs, vecs / norms[norms > TINY,None]])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_BERN_RAD:
        sparse_vecs = jnp.zeros((0,n))

        while len(sparse_vecs) < batch:
            key, subkey = random.split(key)
            bernoulli = random.uniform(subkey, shape=(batch,n))
            entries = jnp.ones((batch,n))/jnp.sqrt(n*eps)
            signed_entries = jnp.where(bernoulli < 0.5, entries, -1*entries)
            vecs = jnp.where(bernoulli < eps, signed_entries, jnp.zeros(signed_entries.shape))
            norms = jnp.linalg.norm(vecs, axis=1) # (batch,)

            vecs = vecs[norms > 10*TINY]
            sparse_vecs = jnp.concatenate([sparse_vecs, vecs / norms[norms > 10*TINY,None]])

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
    cov = (1/n)*jnp.eye(n) if cov is None else cov
    key, subkey1, subkey2, subkey3 = random.split(key, 4)
    v0 = get_sparse_vector(subkey1, n, eps, batch, v0_sampling) # (batch,n)

    V_noise = get_gaussian_vectors(subkey2, n, d-1, batch, cov) # (batch,n,d-1)

    V = jnp.concatenate([v0[...,None], V_noise], axis=-1)
    assert V.shape == (batch, n, d)

    W = get_orthogonal_basis(subkey3, V) # (batch,n,d)

    return v0, W
    