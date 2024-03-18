"""
fn.py
=====

Utilities for setting up custom mean and kernel functions

Created by Maxim Ziatdinov (email: maxim.ziatdinov@gmail.com)
"""

import inspect
import re

from typing import List, Callable, Optional

import jax

from ..kernels.kernels import square_scaled_distance, add_jitter, _sqrt


def set_fn(func: Callable) -> Callable:
    """
    Transforms the given deterministic function to use a params dictionary
    for its parameters, excluding the first one (assumed to be the dependent variable).

    Args:
    - func (Callable): The deterministic function to be transformed.

    Returns:
    - Callable: The transformed function where parameters are accessed
                from a `params` dictionary.
    """
    # Extract parameter names excluding the first one (assumed to be the dependent variable)
    params_names = list(inspect.signature(func).parameters.keys())[1:]

    # Create the transformed function definition
    transformed_code = f"def {func.__name__}(x, params):\n"

    # Retrieve the source code of the function and indent it to be a valid function body
    source = inspect.getsource(func).split("\n", 1)[1]
    source = "    " + source.replace("\n", "\n    ")

    # Replace each parameter name with its dictionary lookup using regex
    for name in params_names:
        source = re.sub(rf'\b{name}\b', f'params["{name}"]', source)

    # Combine to get the full source
    transformed_code += source

    # Define the transformed function in the local namespace
    local_namespace = {}
    exec(transformed_code, globals(), local_namespace)

    # Return the transformed function
    return local_namespace[func.__name__]


def set_kernel_fn(func: Callable,
                  independent_vars: List[str] = ["X", "Z"],
                  jit_decorator: bool = True,
                  docstring: Optional[str] = None) -> Callable:
    """
    Transforms the given kernel function to use a params dictionary for its hyperparameters.
    The resultant function will always add jitter before returning the computed kernel.

    Args:
        func (Callable): The kernel function to be transformed.
        independent_vars (List[str], optional): List of independent variable names in the function. Defaults to ["X", "Z"].
        jit_decorator (bool, optional): @jax.jit decorator to be applied to the transformed function. Defaults to True.
        docstring (Optional[str], optional): Docstring to be added to the transformed function. Defaults to None.

    Returns:
        Callable: The transformed kernel function where hyperparameters are accessed from a `params` dictionary.
    """

    # Extract parameter names excluding the independent variables
    params_names = [k for k, v in inspect.signature(func).parameters.items() if v.default == v.empty]
    for var in independent_vars:
        params_names.remove(var)

    transformed_code = ""
    if jit_decorator:
        transformed_code += "@jit" + "\n"

    additional_args = "noise: int = 0, jitter: float = 1e-6, **kwargs"
    transformed_code += f"def {func.__name__}({', '.join(independent_vars)}, params: Dict[str, jnp.ndarray], {additional_args}):\n"

    if docstring:
        transformed_code += '    """' + docstring + '"""\n'

    source = inspect.getsource(func).split("\n", 1)[1]
    lines = source.split("\n")

    for idx, line in enumerate(lines):
        # Convert all parameter names to their dictionary lookup throughout the function body
        for name in params_names:
            lines[idx] = re.sub(rf'\b{name}\b', f'params["{name}"]', lines[idx])

    # Combine lines back and then split again by return
    modified_source = '\n'.join(lines)
    pre_return, return_statement = modified_source.split('return', 1)

    # Append custom jitter code
    custom_code = f"    {pre_return.strip()}\n    k = {return_statement.strip()}\n"
    custom_code += """
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k
    """

    transformed_code += custom_code

    local_namespace = {"jit": jax.jit}
    exec(transformed_code, globals(), local_namespace)

    return local_namespace[func.__name__]


def _set_noise_kernel_fn(func: Callable) -> Callable:
    """
    Modifies the GPax kernel function to append "_noise" after "k" in dictionary keys it accesses.

    Args:
        func (Callable): Original function.

    Returns:
        Callable: Modified function.
    """

    # Get the source code of the function
    source = inspect.getsource(func)

    # Split the source into decorators, definition, and body
    decorators_and_def, body = source.split("\n", 1)

    # Replace all occurrences of params["k with params["k_noise in the body
    modified_body = re.sub(r'params\["k', 'params["k_noise', body)

    # Combine decorators, definition, and modified body
    modified_source = f"{decorators_and_def}\n{modified_body}"

    # Define local namespace including the jit decorator
    local_namespace = {"jit": jax.jit}

    # Execute the modified source to redefine the function in the provided namespace
    exec(modified_source, globals(), local_namespace)

    # Return the modified function
    return local_namespace[func.__name__]
