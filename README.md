# TensorPolynomials
Library to compute general polynomials from tensors to tensors in JAX. This is the code to reproduce the experiments in the paper https://arxiv.org/abs/2406.01552. 

## Installation

To install, do the following:

- Copy the code `git clone https://github.com/WilsonGregory/TensorPolynomials.git`.
- Navigate to the TensorPolynomials directory `cd TensorPolynomials`.
- Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
- In order to run JAX on a GPU, you will likely need to follow some additional steps detailed in https://github.com/google/jax#installation. You will probably need to know your CUDA version, which can be found with `nvidia-smi` and/or `nvcc --version`.

## Uses

The models are in the models.py file.

### Reproducing Experiments in the paper

- Run the script `sos_spectral.py` to reproduce the synthetic data experiments.
- Run the script `mnist_outliers.py` to reproduce the MNIST experiments.


## Attribution

If you use this code in a published work, please cite:
```
@misc{gregory2024learningequivarianttensorfunctions,
      title={Learning equivariant tensor functions with applications to sparse vector recovery}, 
      author={Wilson G. Gregory and Josué Tonelli-Cueto and Nicholas F. Marshall and Andrew S. Lee and Soledad Villar},
      year={2024},
      eprint={2406.01552},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2406.01552}, 
}
```
