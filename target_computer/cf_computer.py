import numpy as np


def find_digit_1(tensor):
    """
    :param tensor: (k, 64, 64) ndarray
    Возвращает (k, 2) ndarray, содержащий k пар координат, -- позиции, на которых единичка
    """
    return np.argwhere(tensor)[:, 1:]


def compute_octile_distances(p):
    """
    p: ndarray shape (N, 2), где N=100, A[i] = (x_i, y_i).
    Возвращает ndarray shape (N, 64, 64), в котором D[i,j,k] —
    octile-расстояние между точками (j, k) и (x_i, y_i) на i-ой карте.
    """
    jv, kv = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
    jv_ = jv[None, :, :]  # (1, 64, 64)
    kv_ = kv[None, :, :]  # (1, 64, 64)

    x_ = p[:, 0][:, None, None]  # (N, 1, 1)
    y_ = p[:, 1][:, None, None]  # (N, 1, 1)

    dx = np.abs(jv_ - x_)
    dy = np.abs(kv_ - y_)
    return np.maximum(dx, dy) + (np.sqrt(2) - 1) * np.minimum(dx, dy)


def cf(T):
    B, C = T[..., 1], T[..., 2]
    p = find_digit_1(B)
    dist_map = compute_octile_distances(p)
    return np.where(C != 0, dist_map / C, 0.0)
