import time

import jax
import jax.numpy as jnp
import jax.random as random

import tensorpolynomials.data as tpoly_data

def sos_method(S):
    """
    The sum-of-squares method from: https://arxiv.org/pdf/1512.02337.pdf
    args:
        S (jnp.array): an (batch,n,d) array, where n is the ambient dimension, 
        d is number of vectors
    """
    batch,n,d = S.shape

    A = jnp.sum(jax.vmap(
        lambda ai: (jnp.linalg.norm(ai)**2 - d/n)*jnp.tensordot(ai,ai,axes=0),
    )(S.reshape((batch*n,d))).reshape((batch,n,d,d)),axis=1)
    assert A.shape == (batch,d,d)

    _, eigvecs = jnp.linalg.eigh(A) # ascending order
    assert eigvecs.shape == (batch,d,d)
    u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvector
    return jnp.einsum('...ij,...j->...i', S, u)

n = 100
d = 3
eps = 0.2
batch = 20

key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
v0 = tpoly_data.get_sparse_vector(subkey, n, eps, batch) # (batch,n)

key, subkey = random.split(key)
V_noise = tpoly_data.get_gaussian_vectors(subkey, n, d-1, batch) # (batch,n,d-1)
V = jnp.concatenate([v0[...,None], V_noise], axis=-1)
assert V.shape == (batch, n, d)

key, subkey = random.split(key)
W = tpoly_data.get_orthogonal_basis(subkey, V) # (batch,n,d)

out_vec = sos_method(W)
squared_dots = jnp.einsum('...i,...i->...', v0, out_vec)**2
print(jnp.mean(squared_dots))