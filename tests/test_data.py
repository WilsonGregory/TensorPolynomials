import time
import pytest

from jax import random
import jax.numpy as jnp

import tensorpolynomials.data as tpoly_data

TINY = 1.e-5

class TestData:
    """
    Class to test the data generating functions of tensorpolynomials.
    """

    def testGetSparseVector(self):
        key = random.PRNGKey(time.time_ns())
        n = 10
        eps = 0.2
        batch = 5

        sparse_vectors = tpoly_data.get_sparse_vector(key, n, eps, batch)
        assert sparse_vectors.shape == (batch,n)
        # test that they are unit vectors
        assert jnp.allclose(jnp.linalg.norm(sparse_vectors, axis=1), jnp.ones(batch))
        # test that all vectors are sufficiently sparse
        assert jnp.allclose(jnp.sum(sparse_vectors**4, axis=1) >= (1/(eps*n)), jnp.ones(batch))

    def testGetGaussianVectors(self):
        key = random.PRNGKey(time.time_ns())
        n = 10
        num_vecs = 100000
        batch = 3
        true_cov = (1/n)*jnp.eye(n)

        batch_gaussian_vecs = tpoly_data.get_gaussian_vectors(key, n, num_vecs, batch, true_cov)
        assert batch_gaussian_vecs.shape == (batch,n,num_vecs) # after removing the batch dimension
        # test that they are unit vectors
        assert jnp.allclose(jnp.linalg.norm(batch_gaussian_vecs, axis=1), jnp.ones((batch,num_vecs)))

        for gaussian_vecs in batch_gaussian_vecs:
            cov = jnp.cov(gaussian_vecs)
            # assert that the cov is correct
            assert jnp.allclose(cov, true_cov, atol=1e-3, rtol=1e-3), print(jnp.max(jnp.abs(cov - true_cov)))
        
    def testGetOrthogonalBasis(self):
        key = random.PRNGKey(time.time_ns())
        key, subkey1, subkey2 = random.split(key, 3)
        n = 10
        d = 3
        batch = 1
        V = random.normal(subkey1, shape=(batch,n,d))

        W = tpoly_data.get_orthogonal_basis(subkey2, V)[0]
        # assert the columns are orthonormal
        assert jnp.allclose(W.T @ W, jnp.eye(d), rtol=TINY, atol=TINY)

        # assert that the columns of v are in the span of W
        assert jnp.allclose(W @ W.T @ V[0], V[0], rtol=TINY, atol=TINY), jnp.max(jnp.abs(W @ W.T @ V[0] - V[0]))

