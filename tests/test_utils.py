import sys
import numpy as onp
import jax.numpy as jnp
from numpy.testing import assert_equal, assert_

sys.path.insert(0, "../gpax/")

from gpax.utils import preprocess_sparse_image


def test_sparse_img_processing():
    img = onp.random.randn(16, 16)
    # Generate random indices
    idx = [onp.random.randint(0, 16) for _ in range(100)], [onp.random.randint(0, 16) for _ in range(100)]
    # Set these indices to zero
    img[idx] = 0
    # Test the utility function
    X, y, X_full = preprocess_sparse_image(img)
    assert_equal(X.ndim, 2)
    assert_equal(y.ndim,  1)
    assert_equal(X_full.ndim, 2)
    assert_(X.shape[0] < 16*16)
    assert_equal(X.shape[1], 2)
    assert_equal(y.shape[0], X.shape[0])
    assert_equal(X_full.shape[0], 16*16)
    assert_equal(X_full.shape[1], 2)
