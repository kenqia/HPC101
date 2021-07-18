import numpy as np
from numpy import int64


def bilinear_interp_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This is the vectorized implementation of bilinear interpolation.
    - a is a ND array with shape [N, H1, W1, C], dtype = int64
    - b is a ND array with shape [N, H2, W2, 2], dtype = float64
    - return a ND array with shape [N, H2, W2, C], dtype = int64
    """
    # get axis size from ndarray shape
    N, H1, W1, C = a.shape
    N1, H2, W2, _ = b.shape
    assert N == N1

    # TODO: Implement vectorized bilinear interpolation
    idx_C = np.floor(b).astype(int)
    shift = b - idx_C
    shift_x = shift[:, :, :, 0]
    shift_y = shift[:, :, :, 1]
    idx_N = np.arange(N).repeat(H2 * W2)
    idx_H1 = idx_C[:, :, :, 0]
    idx_W1 = idx_C[:, :, :, 1]
    res = a[idx_N, idx_H1.reshape(-1), idx_W1.reshape(-1)].reshape(-1) * ((1 - shift_x) * (1 - shift_y)).repeat(C) + \
          a[idx_N, idx_H1.reshape(-1) + 1, idx_W1.reshape(-1)].reshape(-1) * (shift_x * (1 - shift_y)).repeat(C) + \
          a[idx_N, idx_H1.reshape(-1), idx_W1.reshape(-1) + 1].reshape(-1) * ((1 - shift_x) * shift_y).repeat(C) + \
          a[idx_N, idx_H1.reshape(-1) + 1, idx_W1.reshape(-1) + 1].reshape(-1) * (shift_x * shift_y).repeat(C)
    return res.reshape(N, H2, W2, C).astype(int64)
