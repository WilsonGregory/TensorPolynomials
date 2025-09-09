import math
import numpy as np
import itertools as it

import jax
import jax.numpy as jnp

LETTERS = "abcdefghijklmnopqrstuvwxyxABCDEFGHIJKLMNOPQRSTUVWXYZ"
TINY = 1e-5


def fill_triangular(x, d):
    """
    Constructs an upper triangular matrix from x: https://github.com/google/jax/discussions/10146
    Note that the order is a little funky, for example a 5x5 from 15 values will be:
    array([[ 1,  2,  3,  4,  5],
            [ 0,  7,  8,  9, 10],
            [ 0,  0, 13, 14, 15],
            [ 0,  0,  0, 12, 11],
            [ 0,  0,  0,  0,  6]])
    args:
        x (jnp.array): array of values that we want to become the upper triangular matrix
        d (int): the dimensions of the matrix we are making, d x d
    """
    assert len(x) == (d + d * (d - 1) // 2)
    xc = jnp.concatenate([x, x[d:][::-1]])
    y = jnp.reshape(xc, [d, d])
    return jnp.triu(y, k=0)


def times_group_element(data, gg, precision=jax.lax.Precision.HIGHEST):
    k = data.ndim

    einstr = LETTERS[:k] + ","
    einstr += ",".join([LETTERS[i + 13] + LETTERS[i] for i in range(k)])
    tensor_inputs = (data,) + k * (gg,)
    return jnp.einsum(einstr, *tensor_inputs, precision=precision)


def perm_matrix_from_seq(seq):
    """
    Give a sequence tuple, return the permutation matrix for that sequence
    """
    D = len(seq)
    permutation_matrix = []
    for num in seq:
        row = [0] * D
        row[num] = 1
        permutation_matrix.append(row)
    return np.array(permutation_matrix)


def get_transpositions(n):
    """
    Every permutation is a product of transpositions, so if a tensor is invariant to all
    transpositions, it is invariant to all permutations. Return all transpositions and the identity
    permutation.
    args:
        n (int): sequence length
    """
    transpositions = [list(range(n))]
    for a, b in it.combinations(range(n), 2):
        seq = list(range(n))
        seq[a] = b
        seq[b] = a
        transpositions.append(seq)

    return transpositions


def tensor_from_pattern(k, n, pattern):
    """
    Get a permutation invariant tensor from the specified pattern
    args:
        k (int): tensor order
        n (int): tensor dimension
        pattern (tuple): a tuple that specifies which indices match. E.g. (1,2,1,1) means
        that the 1st, 3rd, and 4th indices match so we only have 2 unique indices.
    """
    num_unique = len(np.unique(pattern))
    assert num_unique <= k
    idxs_arr = np.array(list(it.permutations(range(n), num_unique)))  # (|S_n|,#unique indices)

    flipped_idxs = tuple(tuple(row) for row in idxs_arr.T)

    idxs = tuple(flipped_idxs[i] for i in pattern)

    z = np.zeros((n,) * k)
    z[idxs] = 1

    return jnp.array(z)


def get_symmetric_patterns(patterns, symmetric_axes):
    """
    Remove patterns which are not symmetric.
    args:
        patterns (list of tuples): possible patterns
        symmetric_axes (tuple of tuples of ints): symmetric axes must be pairs
    """
    valid_patterns = []
    paired_patterns = {}
    for pattern in patterns:
        valid = True
        for a, b in symmetric_axes:
            # if they are different, and a or b is tied to another, then it fails
            if pattern[a] != pattern[b]:
                pattern_arr = np.array(pattern)
                if np.sum(pattern_arr == pattern[a]) > 1 or np.sum(pattern_arr == pattern[b]) > 1:
                    # make the swpa pattern
                    pattern_arr[a] = pattern[b]
                    pattern_arr[b] = pattern[a]
                    swap_pattern = tuple(reorder_pattern(pattern_arr))
                    if (pattern not in paired_patterns) and (swap_pattern not in paired_patterns):
                        paired_patterns[pattern] = (pattern, swap_pattern)

                    valid = False

        if valid:
            valid_patterns.append(pattern)

    return valid_patterns, list(paired_patterns.values())


def partitions(n, k, min_elem=0):
    """
    From: https://stackoverflow.com/questions/28965734/general-bars-and-stars
    args:
        n (int): number of items to partition
        k (int): number of partitions
        min_elem (int): minimum number of elements per partition
    """
    n = n - (min_elem * k)
    assert n > -1
    for c in it.combinations(range(n + k - 1), k - 1):
        yield [b - a - 1 + min_elem for a, b in zip((-1,) + c, c + (n + k - 1,))]


def reorder_pattern(pattern):
    """
    Reorder a pattern so that the numbers are in order, e.g. (1,0,0,1) -> (0,1,1,0).
    However, (1,0,0,2) is a valid order because only the singletons can be swapped.
    args:
        partition (tuple of ints): the size of the parititions
        pattern (np.array): the elements of the pattern, as a numpy array
    """
    k = len(pattern)
    _, partition = np.unique(pattern, return_counts=True)
    new_pattern = np.copy(pattern)
    for j in range(1, (k // 2) + 1):
        if np.sum(partition == j) > 1:  # if there are multiple indices that can be swapped
            idxs = np.arange(len(partition))[partition == j]  # get those indices

            ord = {}
            i = 0
            for char in pattern:
                if (char in idxs) and (char not in ord):
                    ord[char] = idxs[i]
                    i += 1

            for k, v in ord.items():
                new_pattern[pattern == k] = v

    return new_pattern


def get_patterns(k, n):
    """
    Get the possible patterns. Returns list of tuples of patterns.
    args:
        k (int): tensor order
        n (int):
    """
    patterns = []
    for i in range(
        1, min(k, n) + 1
    ):  # available digits you can use, can't be 4 unique digits if n=3
        for p in partitions(
            k, i, 1
        ):  # possible partitions of k items in i buckets, min=1 item per bucket
            if (
                sorted(p, reverse=True) != p
            ):  # ensure p is in descending order because (1,3) and (3,1) are the same
                continue

            chars = it.chain.from_iterable(
                [(char,) * char_count for char, char_count in enumerate(p)]
            )
            arr = np.unique(np.array(list(it.permutations(chars))), axis=0)  # (num_permutations,k)

            # for each set of matching number of elements in partition, ensure they are in order
            valid_rows = list((True,) * len(arr))
            for row_i in range(len(arr)):
                if np.any(reorder_pattern(arr[row_i]) != arr[row_i]):
                    valid_rows[row_i] = False

            patterns.append([tuple(row) for row in arr[valid_rows]])

    return list(it.chain.from_iterable(patterns))


class PermInvariantTensor:

    basis_dict = {}

    @classmethod
    def get(cls, k, n, symmetric_axes=()):
        """
        args:
            k (int): the order of the tensor
            n (int): the side length of the tensor
            symmetric_axes (tuple of pairs of ints): which axes need to be symmetric, defaults to ()
        """
        if (k, n, symmetric_axes) not in cls.basis_dict:
            print(f"Constructing invariant tensor {(k,n,symmetric_axes)}... ", end="")
            patterns = get_patterns(k, n)
            patterns, paired_patterns = get_symmetric_patterns(patterns, symmetric_axes)
            perm_invariant_tensors = [tensor_from_pattern(k, n, pattern) for pattern in patterns]
            paired_tensors = [
                tensor_from_pattern(k, n, a) + tensor_from_pattern(k, n, b)
                for a, b in paired_patterns
            ]
            cls.basis_dict[(k, n, symmetric_axes)] = jnp.stack(
                perm_invariant_tensors + paired_tensors
            )
            print(f"shape: {cls.basis_dict[(k,n,symmetric_axes)].shape}")

        return cls.basis_dict[(k, n, symmetric_axes)]


def metric_tensor_r(k: int) -> list[tuple[int, ...]]:
    """
    Generate the distinct permutations of k metric tensors
    """
    assert k % 2 == 0
    if k == 2:
        return [(1, 0)]
    else:
        seqs = metric_tensor_r(k - 2)
        ls = []
        for seq in seqs:
            for idx in range(len(seq) + 1):
                ls.append((k - 1,) + seq[:idx] + (k - 2,) + seq[idx:])

        return ls


def final_permutations(k: int, remaining_k: int, n_initial: int = 0) -> list[tuple[int, ...]]:
    all_permutations = []
    for positions in it.combinations(range(n_initial, k + n_initial), remaining_k):
        seq = list(range(n_initial))

        remaining_k_ls = list(reversed(range(n_initial, n_initial + remaining_k)))
        kron_delta_ls = list(reversed(range(n_initial + remaining_k, n_initial + k)))
        for idx in range(n_initial, n_initial + k):
            seq.append(remaining_k_ls.pop() if idx in positions else kron_delta_ls.pop())

        all_permutations.append(tuple(seq))

    return all_permutations


def B(k: int) -> int:
    assert k % 2 == 0
    return math.factorial(k) // (math.factorial(k // 2) * (2 ** (k // 2)))


def metric_tensor_basis_size(total_k: int, n: int) -> int:
    total = 0
    for k in range(2, total_k + 1):
        for j in range(k // 2):
            n_metric_tensor = j + 1
            total += (
                B(2 * n_metric_tensor)
                * (n ** (k - 2 * n_metric_tensor))
                * math.comb(k, 2 * n_metric_tensor)
            )

    return total
