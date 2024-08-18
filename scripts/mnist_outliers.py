import time
import functools
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import equinox as eqx

import tensorpolynomials.data as tpoly_data
import tensorpolynomials.models as models
import tensorpolynomials.ml as ml

def labels_only(data_labels, labels):
    return functools.reduce(
        lambda carry, label: carry | (data_labels == label), 
        labels, 
        jnp.full_like(data_labels, False, dtype=bool),
    )

def load_mnist(dir, key, labels, train_size, val_size, test_size, normalize_length=True):
    mndata = MNIST(dir)
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = jnp.array(np.array(train_images))
    train_labels = jnp.array(np.array(train_labels))
    test_images = jnp.array(np.array(test_images))
    test_labels = jnp.array(np.array(test_labels))

    valid_train = labels_only(train_labels, labels)
    train_images = train_images[valid_train]
    train_labels = train_labels[valid_train]

    valid_test = labels_only(test_labels, labels)
    test_images = test_images[valid_test]
    test_labels = test_labels[valid_test]

    if normalize_length:
        train_images = train_images / jnp.linalg.norm(train_images, axis=1, keepdims=True)
        test_images = test_images / jnp.linalg.norm(test_images, axis=1, keepdims=True)

    subkey1, subkey2 = random.split(key)
    sigma = random.permutation(subkey1, len(train_images))
    sigma2 = random.permutation(subkey2, len(test_images))

    val_images = train_images[sigma][:val_size]
    val_labels = train_labels[sigma][:val_size]

    train_images = train_images[sigma][val_size:train_size + val_size]
    train_labels = train_labels[sigma][val_size:train_size + val_size]

    test_images = test_images[sigma2][:test_size]
    test_labels = test_labels[sigma2][:test_size]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def normalize_and_noise(data, key, d, noise_type='choice256_rotate', noise_var=None):
    num_points, img_size = data.shape
    key, subkey1, subkey2, subkey3 = random.split(key, 4)

    normalized_data = data / jnp.linalg.norm(data, axis=1, keepdims=True)

    if noise_type == 'choice256_rotate':
        noise = random.choice(subkey1, 256, shape=(num_points,img_size,d-1)) # this could be adapted
        aug_data = jnp.concatenate([normalized_data[...,None], noise], axis=-1) # (b,pixels,views)
    elif noise_type == 'normal_rotate': # columns of normal data, then right multiply by orthogonal matrix
        noise = random.normal(subkey1, shape=(num_points,img_size,d-1))
        aug_data = jnp.concatenate([normalized_data[...,None], noise], axis=-1) # (b,pixels,views)
    elif noise_type == 'normal': # original image plus gaussian noise, d times
        noise_var = 0.01 if noise_var is None else noise_var
        noise = noise_var*random.normal(subkey1, shape=(num_points,img_size,d))
        aug_data = normalized_data[...,None] + noise
    elif noise_type == 'mask': # mask a certain percent of pixels with random values 0-255
        # for this one, use the raw (unormalized) image data
        noise_var = 0.1 if noise_var is None else noise_var # percent of changed pixels
        mask = random.choice(subkey1, 256, shape=(num_points,img_size,d))
        idxs = random.bernoulli(subkey2, noise_var, shape=(num_points,img_size,d))
        aug_data = jnp.where(idxs, mask, data[...,None])
    elif noise_type == 'block':
        sidelen = 28 # currently hardcoded
        noise_var = 8 if noise_var is None else noise_var # sidelength of block

        zed = jnp.stack(jnp.meshgrid(jnp.arange(sidelen),jnp.arange(sidelen), indexing='ij'), axis=-1).reshape((-1,2)) #(28**2,2)

        block_vals = random.choice(subkey1, 256, shape=(len(data),d))
        aug_data = np.zeros(data.shape + (d,))
        for i, img in enumerate(data): # ugh, for loops
            assert img.shape == (28**2,)
            for j in range(d):
                key, subkey = random.split(key)
                idx_x, idx_y = random.choice(subkey, sidelen - noise_var, shape=(2,)) # an x and y coordinate
                valid_x = (zed[:,0] >= idx_x ) & (zed[:,0] < (idx_x + noise_var))
                valid_y = (zed[:,1] >= idx_y ) & (zed[:,1] < (idx_y + noise_var))

                aug_data[i,:,j] = jnp.where(valid_x & valid_y, block_vals[i,j], img)

        aug_data = jnp.array(aug_data)

    aug_data = aug_data / jnp.linalg.norm(aug_data, axis=1, keepdims=True)
    aug_data_orthogonal = tpoly_data.get_orthogonal_basis(subkey3, aug_data) 

    return aug_data_orthogonal, normalized_data, aug_data # (batch,n,d)

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

dir = '/data/wgregor4/mnist'
n = 28 ** 2
d = 20 # problems with having enough memory
width = 128
depth = 2
train_size = 900
val_size = 100
test_size = 100
labels = jnp.arange(10)

train = False
save = False
load_results = False
save_dir = '../runs/sparse_vector/'
trials = 1
verbose = 0
# print_noise_types = '../images/mnist_outliers/noise_types.png'
# print_sample_result = '../images/mnist_outliers/sample_result.png'
print_noise_types = None
print_sample_result = None
table_print = True

key = random.PRNGKey(time.time_ns())
key, subkey = random.split(key)
train_sparse, train_labels, val_sparse, val_labels, test_sparse, test_labels = load_mnist(
    dir, 
    subkey, 
    labels, 
    train_size, 
    val_size, 
    test_size,
    normalize_length=False, # do this later in noise
)

train_size = len(train_sparse)
batch_size = 10

# lrs are currently untuned
key, subkey1, subkey2, subkey3, subkey4 = random.split(key, 5)
models_list = [
    # ('baselineCNN', 'CNN', 1e-3, models.BaselineCNN(d,width,subkey4)),
    ('baselineLearned', 'MLP', 1e-3, models.BaselineLearnedModel(n, d, width, depth, subkey3)), # n*d inputs, d*d outputs
    ('sparseDiagonal', 'SVH-Diag', 5e-4, models.SparseVectorHunterDiagonal(n, width, depth, subkey2)), # n inputs, n+1 outputs
    ('sparse', 'SVH', 3e-4, models.SparseVectorHunter(n, width, depth, subkey3)), # n+(n(n-1)/2) inputs, n+1+(n(n-1)/2) outputs
]

# noise_types = [('choice256_rotate',None), ('normal_rotate',None), ('normal',0.05), ('mask',0.3), ('block', 12)]
noise_types = [
    ('normal_rotate', 'Subspace', None), 
    ('normal', 'Gaussian', 0.05), 
    ('mask', 'Bernoulli', 0.3), 
    ('block', 'Block', 12),
]
for model_name, _, lr, model in models_list:
    print(f'{model_name}: {sum([x.size for x in jax.tree_util.tree_leaves(eqx.filter(model,eqx.is_array))]):,} params')

results = np.zeros((trials,len(noise_types),len(models_list)+2,2))

plt.rcParams['axes.titlesize'] = 40
if print_noise_types is not None:
    ncols = 1 + min(d,3) + min(d,3)
    fig, axes = plt.subplots(len(noise_types),ncols,figsize=(8*ncols,8*len(noise_types)))
    for i, (noise_type, noise_name, noise_var) in enumerate(noise_types):
        test_X, test_Y, nonortho_test_X = normalize_and_noise(test_sparse, subkey1, d, noise_type, noise_var)

        axes[i,0].imshow(test_Y[0].reshape((28,28)), cmap='gray')
        axes[i,0].get_xaxis().set_ticks([])
        axes[i,0].get_yaxis().set_ticks([])
        # axes[i,0].set_title('Target Image')
        axes[i,0].set_aspect('equal')

        for j in range(1,1+min(d,3)):
            axes[i,j].imshow(nonortho_test_X[0,:,j-1].reshape((28,28)), cmap='gray')
            axes[i,j].get_xaxis().set_ticks([])
            axes[i,j].get_yaxis().set_ticks([])
            # axes[i,j].set_title(f'Raw {noise_name} {j}')
            axes[i,j].set_aspect('equal')

        for j in range(1+min(d,3),1+min(d,3)+min(d,3)):
            axes[i,j].imshow(test_X[0,:,j-(1+min(d,3))].reshape((28,28)), cmap='gray')
            axes[i,j].get_xaxis().set_ticks([])
            axes[i,j].get_yaxis().set_ticks([])
            # axes[i,j].set_title(f'Basis {noise_name} {j-min(d,3)}')
            axes[i,j].set_aspect('equal')

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(print_noise_types)
    plt.close()

if print_sample_result is not None:
    ncols = 1 + 2 + len(models_list)
    fig, axes = plt.subplots(len(noise_types),ncols,figsize=(8*ncols,8*len(noise_types)))

if load_results:
    results = jnp.load(f'{save_dir}/sparse_vector_mnist_results_t{trials}_N{train_size}_n{n}_d{d}.npy')
else:
    for t in range(trials):
        for i, (noise_type, noise_name, noise_var) in enumerate(noise_types):
            key, subkey1, subkey2, subkey3 = random.split(key, 4)
            train_X, train_Y, _ = normalize_and_noise(train_sparse, subkey1, d, noise_type, noise_var)
            val_X, val_Y, _ = normalize_and_noise(val_sparse, subkey2, d, noise_type, noise_var)
            test_X, test_Y, noisy_test_X = normalize_and_noise(test_sparse, subkey3, d, noise_type, noise_var)

            results[t,i,0,0] = 1 - map_and_loss(models.sos_method, train_X, train_Y)
            results[t,i,0,1] = 1 - map_and_loss(models.sos_method, test_X, test_Y)
            start_time = time.time()
            results[t,i,0,1] = 1 - map_and_loss(models.sos_method, test_X, test_Y)
            elapsed = time.time() - start_time
            print(f'{t},{noise_type},sos: {results[t,i,0,1]}, time:{elapsed}')

            results[t,i,1,0] = 1 - map_and_loss(models.sos_methodII, train_X, train_Y)
            results[t,i,1,1] = 1 - map_and_loss(models.sos_methodII, test_X, test_Y)
            start_time = time.time()
            results[t,i,1,1] = 1 - map_and_loss(models.sos_methodII, test_X, test_Y)
            elapsed = time.time() - start_time
            print(f'{t},{noise_type},sosII: {results[t,i,1,1]}, time:{elapsed}')

            if (t==0) and (print_sample_result is not None):
                axes[i,0].imshow(test_Y[0].reshape((28,28)), cmap='gray')
                axes[i,0].get_xaxis().set_ticks([])
                axes[i,0].get_yaxis().set_ticks([])
                # axes[i,0].set_title('Target Image')
                axes[i,0].set_aspect('equal')

                axes[i,1].imshow(models.sos_method(test_X[0]).reshape((28,28)), cmap='gray')
                axes[i,1].get_xaxis().set_ticks([])
                axes[i,1].get_yaxis().set_ticks([])
                # axes[i,1].set_title(f'{noise_name}, SoS I')
                axes[i,1].set_aspect('equal')

                axes[i,1+1].imshow(models.sos_methodII(test_X[0]).reshape((28,28)), cmap='gray')
                axes[i,1+1].get_xaxis().set_ticks([])
                axes[i,1+1].get_yaxis().set_ticks([])
                # axes[i,1+1].set_title(f'{noise_name}, SoS II')
                axes[i,1+1].set_aspect('equal')

            for k, (model_name, print_name, lr, model) in enumerate(models_list):

                if train:
                    steps_per_epoch = int(train_size/batch_size)
                    key, subkey = random.split(key)
                    trained_model, _, _ = ml.train(
                        model, 
                        map_and_loss, 
                        train_X, 
                        train_Y, 
                        subkey, 
                        ml.EpochStop(epochs=30, verbose=verbose),
                        optax.adam(optax.exponential_decay(lr, int(train_size/batch_size), 0.999)), 
                        batch_size,
                        val_X,
                        val_Y,
                    )
                    if save:
                        eqx.tree_serialise_leaves(
                            f'{save_dir}{model_name}_t{t}_{noise_type}{noise_var}_N{train_size}_d{d}.eqx',
                            trained_model,
                        )
                else:
                    trained_model = eqx.tree_deserialise_leaves(
                        f'{save_dir}{model_name}_t{t}_{noise_type}{noise_var}_N{train_size}_d{d}.eqx',
                        model,
                    )

                key, subkey1, subkey2 = random.split(key, 3)
                results[t,i,k+2,0] = 1 - ml.map_loss_in_batches(map_and_loss, trained_model, train_X, train_Y, batch_size, subkey1)
                results[t,i,k+2,1] = 1 - ml.map_loss_in_batches(map_and_loss, trained_model, test_X, test_Y, batch_size, subkey2)
                start_time = time.time()
                results[t,i,k+2,1] = 1 - ml.map_loss_in_batches(map_and_loss, trained_model, test_X, test_Y, batch_size, subkey2)
                elapsed = time.time() - start_time

                print(f'{t},{noise_type},{model_name}: {results[t,i,k+2,1]}, time_ns:{elapsed}')

                if (t==0) and (print_sample_result is not None):
                    axes[i,1+2+k].imshow(trained_model(test_X[0]).reshape((28,28)), cmap='gray')
                    axes[i,1+2+k].get_xaxis().set_ticks([])
                    axes[i,1+2+k].get_yaxis().set_ticks([])
                    # axes[i,1+2+k].set_title(f'{noise_name}, {print_name}')
                    axes[i,1+2+k].set_aspect('equal')

        if (t==0) and (print_sample_result is not None):
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(print_sample_result)
            plt.close()

    jnp.save(
        f'{save_dir}/sparse_vector_mnist_results_t{trials}_N{train_size}_n{n}_d{d}.npy',
        results,
    )


mean_results = jnp.mean(results, axis=0)
std_results = jnp.std(results, axis=0)

print_models = [('sos', models.sos_method), ('sosII', models.sos_methodII)] + models_list
if table_print:
    mean_results = jnp.around(mean_results, 3)
    std_results = jnp.around(std_results, 3)

    for l in [0,1]: # train is 0, test is 1
        if l == 0:
            print('Train')
        else:
            print('Test')

        print('\\hline')
        for i, (noise_type, noise_name, noise_var) in enumerate(noise_types):

            print(f'{noise_name} ', end='')

            for k in range(len(print_models)):
                if jnp.allclose(mean_results[i,k,l], jnp.max(mean_results[i,:,l])):
                    print(f'& \\textbf{"{"}{mean_results[i,k,l]:.3f} $\\pm$ {std_results[i,k,l]:.3f}{"}"} ', end='')
                else:
                    print(f'& {mean_results[i,k,l]:.3f} $\\pm$ {std_results[i,k,l]:.3f} ', end='')

            print('\\\\')
            print('\\hline')
        
        print('\n')
else:
    for i, (noise_type, noise_name, noise_var) in enumerate(noise_types):
        print(f'noise_type: {noise_type}')
        print(mean_results[i])
        print(f'+ {std_results[i]}')