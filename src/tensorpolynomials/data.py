from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import signax

import tensorpolynomials.utils as utils

TINY = 1e-5

V0_NORMAL = "Accept/Reject"
V0_BERN_GAUS = "Bernoulli-Gaussian"
V0_BERN_DUB_GAUS = "Bernoulli-Dual-Gaussian"
V0_BERN_RAD = "Bernoulli-Rademacher"
V0_KSPARSE = "v0_ksparse"


def nonzero_and_norm(vecs):
    """
    Take vecs, remove those with 0 norm, then normalize the rest.
    args:
        vecs (jnp.array): vecs of shape (batch,n)
    """
    norms = jnp.linalg.norm(vecs, axis=1)  # (batch,)
    vecs = vecs[norms > TINY]
    return vecs / norms[norms > TINY, None]


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
        print(f"generating {batch} sparse vectors...")
        vecs = []
        with tqdm(total=batch) as pbar:
            while len(vecs) < batch:
                key, subkey = random.split(key)
                vec = random.normal(subkey, shape=(n,))
                vec = vec / jnp.linalg.norm(vec)

                if jnp.sum(vec**4) >= (1 / (eps * n)):
                    vecs.append(vec)
                    pbar.update(1)

        sparse_vecs = jnp.stack(vecs)

    elif v0_sampling == V0_BERN_GAUS:
        sparse_vecs = jnp.zeros((0, n))

        while len(sparse_vecs) < batch:
            key, subkey1, subkey2 = random.split(key, 3)
            entries = random.multivariate_normal(
                subkey1,
                jnp.zeros(n),
                (1 / (n * eps)) * jnp.eye(n),
                shape=(batch,),
            )
            bernoulli = random.uniform(subkey2, shape=(batch, n))
            vecs = jnp.where(bernoulli < eps, entries, jnp.zeros(entries.shape))

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_BERN_DUB_GAUS:
        sparse_vecs = jnp.zeros((0, n))
        assert eps <= (1 / 3)
        q = jnp.sqrt((1 / 3) * (1 - eps) * (1 - 3 * eps))

        while len(sparse_vecs) < batch:
            key, subkey1, subkey2, subkey3 = random.split(key, 4)
            big_entries = random.multivariate_normal(
                subkey1,
                jnp.zeros(n),
                ((eps + q) / (n * eps)) * jnp.eye(n),
                shape=(batch,),
            )
            little_entries = random.multivariate_normal(
                subkey2,
                jnp.zeros(n),
                ((1 - eps - q) / (n * (1 - eps))) * jnp.eye(n),
                shape=(batch,),
            )

            bernoulli = random.uniform(subkey3, shape=(batch, n))
            vecs = jnp.where(bernoulli < eps, big_entries, little_entries)

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_BERN_RAD:
        sparse_vecs = jnp.zeros((0, n))

        while len(sparse_vecs) < batch:
            key, subkey = random.split(key)
            bernoulli = random.uniform(subkey, shape=(batch, n))
            entries = jnp.ones((batch, n)) / jnp.sqrt(n * eps)
            signed_entries = jnp.where(bernoulli < 0.5, entries, -1 * entries)
            vecs = jnp.where(bernoulli < eps, signed_entries, jnp.zeros(signed_entries.shape))

            normed_vecs = nonzero_and_norm(vecs)
            sparse_vecs = jnp.concatenate([sparse_vecs, normed_vecs])

        sparse_vecs = sparse_vecs[:batch]

    elif v0_sampling == V0_KSPARSE:
        key, subkey1, subkey2 = random.split(key, 3)
        num_nonzero = np.floor(eps * n).astype(int)  # exact number of nonzero entries per vector

        random.normal(subkey1, shape=((batch, num_nonzero)))
        entries = jnp.concatenate(
            [
                random.normal(subkey1, shape=(batch, num_nonzero)),
                jnp.zeros((batch, n - num_nonzero)),
            ],
            axis=1,
        )
        shuffled_entries = random.permutation(subkey2, entries, axis=1, independent=True)
        sparse_vecs = shuffled_entries / jnp.linalg.norm(shuffled_entries, axis=1, keepdims=True)

    return sparse_vecs


def get_gaussian_vectors(key, n, num_vecs, batch, cov):
    V = random.multivariate_normal(key, jnp.zeros(n), cov, shape=(batch, num_vecs)).transpose(
        (0, 2, 1)
    )
    assert V.shape == (batch, n, num_vecs)
    norms = jnp.linalg.norm(V, axis=1, keepdims=True)  # (batch,1,num_vecs)
    assert norms.shape == (batch, 1, num_vecs)
    return V / norms


def get_orthogonal_basis(key, V):
    """
    args:
        key (rand key): we want a random orthogonal basis
        V (jnp.array): shape (batch,n,d)
    """
    batch, n, d = V.shape
    O = random.orthogonal(key, d, shape=(batch,))
    assert O.shape == (batch, d, d)
    Q, _ = jnp.linalg.qr(jnp.einsum("...ij,...jk->...ik", V, O))
    assert Q.shape == (batch, n, d)
    return Q


def get_synthetic_data(key, n, d, eps, batch, v0_sampling=V0_NORMAL, cov=None):
    isnan = True
    while isnan:
        cov = (1 / n) * jnp.eye(n) if cov is None else cov
        key, subkey1, subkey2, subkey3 = random.split(key, 4)
        v0 = get_sparse_vector(subkey1, n, eps, batch, v0_sampling)  # (batch,n)

        V_noise = get_gaussian_vectors(subkey2, n, d - 1, batch, cov)  # (batch,n,d-1)

        V = jnp.concatenate([v0[..., None], V_noise], axis=-1)
        assert V.shape == (batch, n, d)

        W = get_orthogonal_basis(subkey3, V)  # (batch,n,d)

        # Sometimes this is nan, I am not sure why. It also breaks the other models when
        # this happens. I thought it might be some issue with normalizing/dividing by zero
        # but it keeps happening. So for now, we will just do this.
        isnan = jnp.isnan(1 - map_and_loss(sos_method, W, v0))

    return v0, W


def sos_method(S):
    """
    The sum-of-squares method from: https://arxiv.org/pdf/1512.02337.pdf
    args:
        S (jnp.array): an (n,d) array, where n is the ambient dimension, d is number of vectors
    """
    n, d = S.shape

    vmap_inner = jax.vmap(
        lambda ai: (jnp.linalg.norm(ai) ** 2 - d / n) * jnp.tensordot(ai, ai, axes=0)
    )
    A = jnp.sum(vmap_inner(S), axis=0)
    assert A.shape == (d, d)

    _, eigvecs = jnp.linalg.eigh(A)  # ascending order
    assert eigvecs.shape == (d, d)
    u = eigvecs[..., -1]  # the eigenvector corresponding to the top eigenvalue
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
    squared_dots = jnp.einsum("...i,...i->...", y, pred_y) ** 2
    return 1 - jnp.mean(squared_dots)


## Signature Tensor ish


def expand_signature(
    D: int, signature_flat: jax.Array, orders: list[int] | None = None
) -> dict[int, jax.Array]:
    """
    Given a flat signature, expand it into list form. Signature starts at k=1

    args:
        D: dimension
        signature_flat: shape (batch,flat_signature)
    """
    idx = 0
    signature = {}
    if orders is None:  # assume its orders 1...k
        k = 1
        while idx < signature_flat.shape[1]:
            signature[k] = signature_flat[:, idx : idx + D**k].reshape((-1,) + (D,) * k)
            idx += D**k
            k += 1

        assert idx == signature_flat.shape[1]
    else:
        for k in orders:
            signature[k] = signature_flat[:, idx : idx + D**k].reshape((-1,) + (D,) * k)
            idx += D**k

    return signature


def flatten_signature(signature: dict[int, jax.Array], batch_dims: int = 1) -> jax.Array:
    """
    Given a signature in list form, flatten it into a single array, possibly with leading batch
    dimensions.

    args:
        signature: as a list, shape (batch1, ... batchDim,tensor)
        batch_dims: number of leading batch dimension of each signature element

    return:
        flattened signature
    """
    return jnp.concatenate(
        [t.reshape(t.shape[:batch_dims] + (-1,)) for t in signature.values()], axis=-1
    )


def get_signature_flat(
    D: int,
    n_curves: int,
    sig_order: int,
    top_sig_only: bool,
    steps: int,
    interval: tuple[int, int],
    library_params: tuple[str, int],
    key: ArrayLike,
    coeffs_dist: str = "uniform",
    integrator_steps: int = 1000,
) -> tuple[jax.Array, jax.Array]:
    curves, signature = get_signature(
        D,
        n_curves,
        sig_order,
        top_sig_only,
        steps,
        interval,
        library_params,
        key,
        coeffs_dist,
        integrator_steps,
    )
    return curves, flatten_signature(signature)


def psi(t: jax.Array, i: jax.Array, m: int) -> jax.Array:
    latter_two = jnp.where(
        t <= i / m,
        m * t - (i - 1),
        jnp.ones_like(t),
    )
    assert isinstance(latter_two, jax.Array)

    return jnp.where(t <= (i - 1) / m, jnp.zeros_like(t), latter_two)


def get_signature(
    D: int,
    n_curves: int,
    sig_order: int,
    top_sig_only: int,
    steps: int,
    interval: tuple[int, int],
    library_params: tuple[str, int],
    key: ArrayLike,
    coeffs_dist: str = "uniform",
    integrator_steps: int = 1000,
) -> tuple[jax.Array, dict[int, jax.Array]]:
    start, end = interval
    library_type, degree = library_params

    t = jnp.linspace(start, end, num=steps)  # (steps,))

    # (batch,D,degree)
    if coeffs_dist == "uniform":
        coeffs = random.uniform(key, shape=(n_curves, D, degree), minval=-1)
    elif (
        coeffs_dist == "normal"
    ):  # TODO: I suspect these curves are too smooth for us to beat signax
        coeffs = random.normal(key, shape=(n_curves, D, degree))
    else:
        raise ValueError()

    if library_type == "polynomial":
        library = jnp.stack([t**m for m in range(1, degree + 1)], axis=-1)  # (steps, degree)

        if (sig_order <= 3) and (interval == (0, 1)):
            C1 = jnp.ones(degree)
            sig1 = jnp.einsum("bIi,i->bI", coeffs, C1)

            i2 = jnp.full((degree,) * 2, jnp.arange(degree)[:, None] + 1)
            j2 = jnp.transpose(i2, (1, 0))
            C2 = j2 / (i2 + j2)
            sig2 = jnp.einsum("...Ii,...Jj,ij->...IJ", coeffs, coeffs, C2)

            i3 = jnp.full((degree,) * 3, jnp.arange(degree)[:, None, None] + 1)
            j3 = jnp.transpose(i3, (1, 0, 2))
            k3 = jnp.transpose(i3, (2, 1, 0))
            C3 = (j3 / (i3 + j3)) * (k3 / (i3 + j3 + k3))
            sig3 = jnp.einsum("...Ii,...Jj,...Kk,ijk->...IJK", coeffs, coeffs, coeffs, C3)
            signature = {1: sig1, 2: sig2, 3: sig3}
        else:
            # do it numerically
            t_integrator = jnp.linspace(start, end, num=integrator_steps)
            library_integrator = jnp.stack([t_integrator**m for m in range(1, degree + 1)], axis=-1)
            # (int_steps,degree),(b,D,degree)->(b,int_steps,D)
            curves_integrator = jnp.einsum("ij,bDj->biD", library_integrator, coeffs)
            signature_ls = signax.signature(curves_integrator, sig_order, flatten=False)
            signature = {k: sig for k, sig in zip(range(1, sig_order + 1), signature_ls)}

    elif library_type == "piecewise_linear":
        library = jax.vmap(psi, in_axes=(None, 0, None), out_axes=1)(
            t, jnp.arange(1, degree + 1), degree
        )

        if (sig_order <= 3) and (interval == (0, 1)):
            # we have exact calculation for this case
            C1 = jnp.ones(degree)
            sig1 = jnp.einsum("bIi,i->bI", coeffs, C1)

            # smarter way of doing this no doubt
            C2 = np.zeros((degree, degree))
            for i in range(degree):
                for j in range(degree):
                    if i < j:
                        C2[i, j] = 1
                    elif i == j:
                        C2[i, j] = 0.5
                    # else it equals 0

            sig2 = jnp.einsum("...Ii,...Jj,ij->...IJ", coeffs, coeffs, C2)

            # smarter way of doing this no doubt
            C3 = np.zeros((degree, degree, degree))
            for i in range(degree):
                for j in range(degree):
                    for k in range(degree):
                        if i < j < k:
                            C3[i, j, k] = 1
                        elif (i < j and j == k) or (i == j and j < k):
                            C3[i, j, k] = 0.5
                        elif i == j == k:
                            C3[i, j, k] = 1 / 6
                        # else it equals 0

            sig3 = jnp.einsum("...Ii,...Jj,...Kk,ijk->...IJK", coeffs, coeffs, coeffs, C3)
            signature = {1: sig1, 2: sig2, 3: sig3}
        else:
            # do it numerically
            t_integrator = jnp.linspace(start, end, num=integrator_steps)
            library_integrator = jax.vmap(psi, in_axes=(None, 0, None), out_axes=1)(
                t_integrator, jnp.arange(1, degree + 1), degree
            )
            # (int_steps,degree),(b,D,degree)->(b,int_steps,D)
            curves_integrator = jnp.einsum("ij,bDj->biD", library_integrator, coeffs)
            signature_ls = signax.signature(curves_integrator, sig_order, flatten=False)
            signature = {k: sig for k, sig in zip(range(1, sig_order + 1), signature_ls)}

    else:
        raise ValueError()

    # (steps,degree),(b,D,steps)->(b,steps,D)
    curves = jnp.einsum("ij,bDj->biD", library, coeffs)
    return curves, {sig_order: signature[sig_order]} if top_sig_only else signature


def get_signature_old(
    D: int,
    n_curves: int,
    integrator_steps: int,
    sig_order: int,
    subsample_steps: int,
    key: ArrayLike,
) -> tuple[jax.Array, jax.Array]:
    t = jnp.linspace(-1, 1, num=integrator_steps)  # (steps,))
    library = jnp.stack([t, t**2, t**3, t**4, t**5], axis=-1)

    key, subkey = random.split(key)
    # (batch,library,channels)
    coeffs = random.uniform(subkey, shape=(n_curves, library.shape[1], D), minval=-1)
    curves = jnp.einsum("ij,bjc->bic", library, coeffs)  # (batch,steps,channels)

    signature = signax.signature(curves, sig_order, flatten=True)

    return curves[:, :: (integrator_steps // subsample_steps)], signature
