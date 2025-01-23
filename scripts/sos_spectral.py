import time
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
import optax

import tensorpolynomials.data as tpoly_data
import tensorpolynomials.ml as ml
import tensorpolynomials.models as models

RAND_COV = "Random"
DIAG_COV = "Diagonal"
UNIT_COV = "Identity"

sampling_print_names = {
    tpoly_data.V0_NORMAL: "A/R",
    tpoly_data.V0_BERN_GAUS: "BG",
    tpoly_data.V0_BERN_DUB_GAUS: "CBG",
    tpoly_data.V0_BERN_RAD: "BR",
    tpoly_data.V0_KSPARSE: "KS",
}


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


# Main
key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
train = True
save = False
save_dir = "../runs/"
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
trials = 5
verbose = 0

key, subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, 6)
models_list = [
    (
        "baselineLearned",
        1e-3,
        models.BaselineLearnedModel(n, d, width, depth, subkey1),
    ),  # n*d inputs, d*d outputs
    (
        "sparseDiagonal",
        5e-4,
        models.SparseVectorHunterDiagonal(n, width, depth, subkey2),
    ),  # n inputs, n+1 outputs
    (
        "sparse",
        3e-4,
        models.SparseVectorHunter(n, width, depth, subkey3),
    ),  # n+(n(n-1)/2) inputs, n+1+(n(n-1)/2) outputs
    # ('sparsePerm', 1e-4, models.SVHPerm(n, 32, depth*2, subkey4)),
    # ('direct', 3e-4, models.Direct(n, d, width, depth, subkey5)), # n+(n(n-1)/2) inputs, n outputs
    # ('directDiagonal', 5e-3, models.DirectDiagonal(n, d, width, depth, subkey5)), # n inputs, n outputs
]
for model_name, lr, model in models_list:
    print(f"{model_name}: {sum([x.size for x in jax.tree_util.tree_leaves(model)]):,} params")

samplings = [
    tpoly_data.V0_NORMAL,
    tpoly_data.V0_BERN_GAUS,
    tpoly_data.V0_BERN_DUB_GAUS,
    tpoly_data.V0_BERN_RAD,
]
covariances = [RAND_COV, DIAG_COV, UNIT_COV]

results = np.zeros((trials, len(samplings), len(covariances), len(models_list) + 2, 2))

if train:
    for t in range(trials):
        for i, v0_sampling in enumerate(samplings):
            for j, cov_type in enumerate(covariances):
                key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)

                if cov_type == RAND_COV:
                    # Random covariance matrix, force it to be pos. def.
                    M = random.normal(subkey4, shape=(n, n))
                    cov = M.T @ M + (1e-5) * jnp.eye(n)
                elif cov_type == DIAG_COV:
                    # Random diagonal covariance matrix
                    cov = jnp.diag(random.uniform(subkey, shape=(n,), minval=0.5, maxval=1.5))
                elif cov_type == UNIT_COV:
                    # No covariance, use 1/n * Id
                    cov = None
                else:
                    raise Exception(f"No covariance {cov}")

                # (batch,n), (batch,n,d)
                train_v0, train_W = tpoly_data.get_synthetic_data(
                    subkey1, n, d, eps, train_size, v0_sampling, cov
                )
                val_v0, val_W = tpoly_data.get_synthetic_data(
                    subkey2, n, d, eps, val_size, v0_sampling, cov
                )
                test_v0, test_W = tpoly_data.get_synthetic_data(
                    subkey3, n, d, eps, test_size, v0_sampling, cov
                )

                results[t, i, j, 0, 0] = 1 - map_and_loss(models.sos_method, train_W, train_v0)
                results[t, i, j, 0, 1] = 1 - map_and_loss(models.sos_method, test_W, test_v0)
                print(f"{t},{v0_sampling},{cov_type},sos: {results[t,i,j,0,1]}")

                results[t, i, j, 1, 0] = 1 - map_and_loss(models.sos_methodII, train_W, train_v0)
                results[t, i, j, 1, 1] = 1 - map_and_loss(models.sos_methodII, test_W, test_v0)
                print(f"{t},{v0_sampling},{cov_type},sosII: {results[t,i,j,1,1]}")

                for k, (model_name, lr, model) in enumerate(models_list):
                    steps_per_epoch = int(train_size / batch_size)
                    key, subkey = random.split(key)
                    trained_model, _, _ = ml.train(
                        model,
                        map_and_loss,
                        train_W,
                        train_v0,
                        subkey,
                        ml.ValLoss(patience=20, verbose=verbose),
                        optax.adam(
                            optax.exponential_decay(lr, int(train_size / batch_size), 0.999)
                        ),
                        batch_size,
                        val_W,
                        val_v0,
                    )

                    results[t, i, j, k + 2, 0] = 1 - map_and_loss(trained_model, train_W, train_v0)
                    results[t, i, j, k + 2, 1] = 1 - map_and_loss(trained_model, test_W, test_v0)
                    print(f"{t},{v0_sampling},{cov_type},{model_name}: {results[t,i,j,k+2,1]}")

                if save:
                    jnp.save(
                        f"{save_dir}/sparse_vector_results_t{trials}_N{train_size}_n{n}_d{d}_eps{eps}.npy",
                        results,
                    )
else:
    results = jnp.load(
        f"{save_dir}/sparse_vector_results_t{trials}_N{train_size}_n{n}_d{d}_eps{eps}.npy"
    )

mean_results = jnp.mean(results, axis=0)
std_results = jnp.std(results, axis=0)
print_models = [("sos", models.sos_method), ("sosII", models.sos_methodII)] + models_list
if table_print:
    mean_results = jnp.around(mean_results, 3)
    std_results = jnp.around(std_results, 3)

    for l in [0, 1]:  # train is 0, test is 1
        if l == 0:
            print("Train")
        else:
            print("Test")

        print("\\hline")
        for i, v0_sampling in enumerate(samplings):
            for j, cov_type in enumerate(covariances):
                if j == (len(covariances) // 2):
                    print(f"{sampling_print_names[v0_sampling]} ", end="")

                print(f"& {cov_type} ", end="")

                for k in range(len(print_models)):
                    if jnp.allclose(mean_results[i, j, k, l], jnp.max(mean_results[i, j, :, l])):
                        print(
                            f'& \\textbf{"{"}{mean_results[i,j,k,l]:.3f} $\\pm$ {std_results[i,j,k,l]:.3f}{"}"} ',
                            end="",
                        )
                    else:
                        print(
                            f"& {mean_results[i,j,k,l]:.3f} $\\pm$ {std_results[i,j,k,l]:.3f} ",
                            end="",
                        )

                print("\\\\")

            print("\\hline")

        print("\n")
else:
    for i, v0_sampling in enumerate(samplings):
        print(f"\nv0_sampling: {v0_sampling}")
        for j, cov_type in enumerate(covariances):
            print(f"cov_type: {cov_type}")
            print(mean_results[i, j])
            print(f"+ {std_results[i,j]}")
