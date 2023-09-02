"""
mtkernels.py
==========

Multi-task kernel functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@ai4microscopy.com)
"""


from typing import Dict, Callable

import jax.numpy as jnp
from jax import vmap

from .kernels import add_jitter, get_kernel

kernel_fn_type = Callable[[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jnp.ndarray],  jnp.ndarray]

# Helper function to generate in_axes dictionary
get_in_axes = lambda data: ({key: 0 if key != "noise" else None for key in data.keys()},)


def index_kernel(indices1, indices2, params):
    r"""
    Computes the task kernel matrix for given task indices.
    The task covariance between two discrete indices i and j
    is calculated as:

    .. math::
        task\_kernel_values[i, j] = WW^T[i, j] + v[i] \delta_{ij}

    where :math:`WW^T` is the matrix product of :math:`B` with its transpose, :math:`v[i]`
    is the variance of task :math:`i`, and :math:`\delta_{ij}` is the Kronecker delta
    which is 1 if :math:`i == j` and 0 otherwise.

    Args:
        indices1:
            An array of task indices for the first set of data points.
            Each entry is an integer that indicates the task associated
            with a data point.
        indices2:
            An array of task indices for the second set of data points.
            Each entry is an integer that indicates the task associated
            with a data point.
        params:
            Dictionary of parameters for the task kernel. It includes:
            'W': The coregionalization matrix of shape (num_tasks, num_tasks).
                This is a symmetric positive  semi-definite matrix that determines
                the correlation structure between the tasks.
            'v':
                The vector of task variances with the (n_tasks,) shape.
                This is a diagonal matrix that  determines the variance of each task.

    Returns:
        Computed kernel matrix of the shape (len(indices1), len(indices2)).
        Each entry task_kernel_values[i, j] is the covariance between the tasks
        associated with data point i in indices1 and data point j in indices2.
    """
    W = params["W"]
    v = params["v"]
    B = jnp.dot(W, W.T) + jnp.diag(v)
    return B[jnp.ix_(indices1, indices2)]


def MultitaskKernel(base_kernel, **kwargs1):
    r"""
    Constructs a multi-task kernel given a base data kernel.
    The multi-task kernel is defined as

    .. math::
        K(x_i, y_j) = k_{data}(x, y) * k_{task}(i, j)

    where *x* and *y* are data points and *i* and *j* are the tasks
    associated with these points. The task indices are passed as the
    last column in the input data vectors.

    Args:
        base_kernel:
            The name of the base data kernel or a function that computes
            the base data kernel. This kernel is used to compute the
            similarities in the input space. The built-in kernels are 'RBF',
            'Matern', 'Periodic', and 'NNGP'.

        **kwargs1:
            Additional keyword arguments to pass to the `get_kernel`
            function when constructing the base data kernel.

    Returns:
        The constructed multi-task kernel function.
    """

    data_kernel = get_kernel(base_kernel, **kwargs1)

    def multi_task_kernel(X, Z, params, noise=0, **kwargs2):
        """
        Computes multi-task kernel matrix, given two input arrays and
        a dictionary wuth kernel parameters. The input arrays must have the
        shape (N, D+1) where N is the number of data points and D is the feature
        dimension. The last column contains task indices.
        """

        # Extract input data and task indices from X and Z
        X_data, indices_X = X[:, :-1], X[:, -1].astype(int)
        Z_data, indices_Z = Z[:, :-1], Z[:, -1].astype(int)

        # Compute data and task kernels
        k_data = data_kernel(X_data, Z_data, params, 0, **kwargs2) # noise will be added later
        k_task = index_kernel(indices_X, indices_Z, params)

        # Compute the multi-task kernel
        K = k_data * k_task

        # Add noise associated with each task
        if X.shape == Z.shape:
            # Get the noise corresponding to each sample's task
            if isinstance(noise, (int, float)):
                noise = jnp.ones(1) * noise
            sample_noise = noise[indices_X]
            # Add small jitter for numerical stability
            sample_noise = add_jitter(sample_noise, **kwargs2)
            # Add the noise to the diagonal of the kernel matrix
            K = K.at[jnp.diag_indices(K.shape[0])].add(sample_noise)

        return K

    return multi_task_kernel


def MultivariateKernel(base_kernel, num_tasks, **kwargs1):
    r"""
    Construct a multivariate kernel given a base data kernel asssuming
    that all tasks share the same input space. For situations where not all
    tasks share the same input parameters, see MultitaskKernel.
    The multivariate kernel is defined as a Kronecker product between
    data and task kernels

    .. math::
        K(x_i, y_j) = k_{data}(x, y) \otimes k_{task}(i, j)

    where *x* and *y* are data points and *i* and *j* are the tasks
    associated with these points.

    Args:
        base_kernel:
            The name of the base data kernel or a function that computes
            the base data kernel. This kernel is used to compute the
            similarities in the input space. THe built-in kernels are 'RBF',
            'Matern', 'Periodic', and 'NNGP'.
        num_tasks:
            number of tasks

        **kwargs1 : dict
            Additional keyword arguments to pass to the `get_kernel`
            function when constructing the base data kernel.

    Returns:
        The constructed multi-task kernel function.
    """

    data_kernel = get_kernel(base_kernel, **kwargs1)

    def multivariate_kernel(X, Z, params, noise=0, **kwargs2):
        """
        Computes multivariate kernel matrix, given two input arrays and
        a dictionary wuth kernel parameters. The input arrays must have the
        shape (N, D) where N is the number of data points and D is the feature
        dimension.
        """

        # Compute data and task kernels
        task_labels = jnp.arange(num_tasks)
        k_data = data_kernel(X, Z, params, 0, **kwargs2)  # noise will be added later
        k_task = index_kernel(task_labels, task_labels, params)

        # Compute the multi-task kernel
        K = jnp.kron(k_data, k_task)

        # Add noise associated with each task
        if X.shape == Z.shape:
            # Make sure noise is a jax ndarray with a proper shape
            if isinstance(noise, (float, int)):
                noise = jnp.ones(num_tasks) * noise
            # Add small jitter for numerical stability
            noise = add_jitter(noise, **kwargs2)
            # Create a block-diagonal noise matrix with the noise terms
            # on the diagonal of each block
            noise_matrix = jnp.kron(jnp.eye(k_data.shape[0]), jnp.diag(noise))
            # Add the noise to the diagonal of the kernel matrix
            K += noise_matrix

        return K

    return multivariate_kernel


def LCMKernel(base_kernel, shared_input_space=True, num_tasks=None, **kwargs1):
    """
    Construct kernel for a Linear Model of Coregionalization (LMC)

    Args:
        base_kernel:
            The name of the data kernel or a function that computes
            the data kernel. This kernel is used to compute the
            similarities in the input space. The built-in kernels are 'RBF',
            'Matern', 'Periodic', and 'NNGP'.
        shared_input_space:
            If True (default), assumes that all tasks share the same input space and
            uses a multivariate kernel (Kronecker product).
            If False, assumes that different tasks have different number of observations
            and uses a multitask kernel (elementwise multiplication). In that case, the task
            indices must be appended as the last column of the input vector.
        num_tasks: int, optional
            Number of tasks. This is only used if `shared_input_space` is True.
        **kwargs1:
            Additional keyword arguments to pass to the `get_kernel`
            function when constructing the base data kernel.

    Returns:
        The constructed LMC kernel function.
    """

    if shared_input_space:
        multi_kernel = MultivariateKernel(base_kernel, num_tasks, **kwargs1)
    else:
        multi_kernel = MultitaskKernel(base_kernel, **kwargs1)

    def lcm_kernel(X, Z, params, noise=0, **kwargs2):
        axes = get_in_axes(params)
        k = vmap(lambda p: multi_kernel(X, Z, p, noise, **kwargs2), in_axes=axes)(params)
        return k.sum(0)

    return lcm_kernel
