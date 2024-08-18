import jax
import jax.random as random
import jax.numpy as jnp
import equinox as eqx

import tensorpolynomials.ml as ml
import tensorpolynomials.utils as utils


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
        upper_triangular = utils.fill_triangular(out, d)
        # subtract the diagonal because it will get added twice otherwise
        A = upper_triangular + upper_triangular.T - jnp.diag(upper_triangular)
        assert A.shape == (d,d)

        _, eigvecs = jnp.linalg.eigh(A) # ascending order
        assert eigvecs.shape == (d,d)
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvalue
        return S @ u

class BaselineCNN(eqx.Module):
    conv_layers: list
    fc_layers: list

    def __init__(self, d, width, key):
        out_features = d + (d * (d-1) // 2) # output array needs to be symmetric
        subkey1, subkey2, subkey3, subkey4 = random.split(key, 4)

        # self.layers = [
        #     eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
        #     eqx.nn.MaxPool2d(kernel_size=2),
        #     jax.nn.relu,
        #     jnp.ravel,
        #     eqx.nn.Linear(1728, 512, key=key2),
        #     jax.nn.sigmoid,
        #     eqx.nn.Linear(512, 64, key=key3),
        #     jax.nn.relu,
        #     eqx.nn.Linear(64, 10, key=key4),
        #     jax.nn.log_softmax,
        # ]

        self.conv_layers = [
            eqx.nn.Conv2d(in_channels=d, out_channels=width, kernel_size=5, key=subkey1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(width, width, kernel_size=5, key=subkey2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.fc_layers = [
            eqx.nn.Linear(width*4*4, 50, key=subkey3),
            jax.nn.relu,
            eqx.nn.Linear(50, out_features, key=subkey4)
        ]
        
    def __call__(self, S):
        n,d = S.shape
        assert n == (28**2)
        x = S.T.reshape((d,28,28))

        for layer in self.conv_layers:
            x = layer(x)

        x = x.reshape(-1)

        for layer in self.fc_layers:
            x = layer(x)

        upper_triangular = utils.fill_triangular(x, d)
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

class SVHPerm(eqx.Module):
    layers: list
    last_layer_pairs: eqx.Module
    last_layer_identity: eqx.Module

    def __init__(self, n, width, num_hidden_layers, key):
        """
        Constructor for SVHPerm, parameterizes full function from vectors to 2-tensor.
        args:
            n (int): number of input vectors
            width (int): width of the NN layers
            num_hidden_layers (int): number of hidden layers, input/output of width
            key (rand key): the random key
        """
        basis_sym_nonsym = utils.PermInvariantTensor.get(4,n,((0,1),))
        basis_nonsym_nonsym = utils.PermInvariantTensor.get(4,n)
        basis_nonsym_sym = basis_sym_nonsym.transpose((0,3,4,1,2))
        bias_basis = utils.PermInvariantTensor.get(2,n)

        key, subkey = random.split(key)
        self.layers = [ml.GeneralLinear(basis_sym_nonsym,1,width,True,bias_basis,subkey)]
        for _ in range(num_hidden_layers):
            key, subkey = random.split(key)
            self.layers.append(
                ml.GeneralLinear(basis_nonsym_nonsym,width,width,True,bias_basis,key=subkey),
            )

        key, subkey1, subkey2 = random.split(key, 3)
        self.last_layer_pairs = ml.GeneralLinear(basis_nonsym_sym,width,1,True,bias_basis,subkey1)
        self.last_layer_identity = ml.GeneralLinear(bias_basis,width,1,True,jnp.ones((1,1)),subkey2)

    def __call__(self, S):
        """
        args:
            S (jnp.array): (n,d) array, d vectors in R^n, but we treat them as n vectors in R^d
        """
        n,d = S.shape
        X = (S @ S.T)[None]
        assert X.shape == (1,n,n)

        for layer in self.layers:
            X = jax.nn.leaky_relu(layer(X))

        pairs = self.last_layer_pairs(X)
        assert pairs.shape == (1,n,n), f'{pairs.shape}'
        identity_scalar = self.last_layer_identity(X)
        
        # Now get a basis consisting of outer products of all pairs of rows.
        pairs_basis = jnp.einsum('ab,cd->acbd', S, S)
        assert pairs_basis.shape == (n,n,d,d)

        A = jnp.einsum('ab,abcd->cd', pairs[0], pairs_basis) + identity_scalar * jnp.eye(d)

        _, eigvecs = jnp.linalg.eigh(A) # ascending order
        assert eigvecs.shape == (d,d)
        u = eigvecs[...,-1] # the eigenvector corresponding to the top eigenvector
        return S @ u

class Direct(eqx.Module):
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
        in_features = n + (n*(n-1)//2) # input is pairwise inner products
        out_features = n # output is n scalars

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
        assert out.shape == (n,)
        return out / jnp.linalg.norm(out)

class DirectDiagonal(eqx.Module):
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
        in_features = n # input is pairwise inner products
        out_features = n # output is n scalars

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
        assert out.shape == (n,)
        return out / jnp.linalg.norm(out)