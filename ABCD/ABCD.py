import numpy as np
from tqdm import tqdm
from dijkstra.backwards_dijkstra import dijkstra_single_map


def create_tensor_B(A):
    """
    Функция принимает на вход тензор map размером (N, 64, 64),
    где каждая карта состоит из 0 и 1, и возвращает тензор B
    того же размера, где каждая карта содержит одну случайную
    единицу на месте из исходных единиц в A.

    Параметры:
        map (np.ndarray): Входной тензор размером (N, 64, 64)

    Возвращает:
        np.ndarray: Тензор B размером (N, 64, 64)
    """
    B = np.zeros_like(A, dtype=np.float32)
    for i in range(A.shape[0]):
        ones_indices = np.argwhere(A[i] == 1)
        j, k = ones_indices[np.random.choice(len(ones_indices))]
        B[i, j, k] = 1
    return B


def create_tensor_C(A, B):
    """
    Функция создает тензор C, используя алгоритм обратного Дейкстры.

    Параметры:
        A (np.ndarray): Тензор препятствий размером (N, 64, 64).
        B (np.ndarray): Тензор начальных точек размером (N, 64, 64).

    Возвращает:
        np.ndarray: Тензор C размером (N, 64, 64) с расстояниями.
    """
    N, rows, cols = A.shape
    C = np.full_like(A, np.inf, dtype=np.float32)

    for i in tqdm(range(N), desc="Dijkstra calculation"):
        start = np.argwhere(B[i] == 1)
        start = tuple(start[0])
        C[i] = dijkstra_single_map(A[i], start)

    return C


def create_tensor_D(C):
    """
    Создает тензор D, где каждая карта имеет случайную точку из топ-30%
    самых труднодостижимых точек в карте C.

    Параметры:
        C (np.ndarray): Тензор расстояний размером (N, 64, 64).

    Возвращает:
        np.ndarray: Тензор D размером (N, 64, 64) с начальной точкой.
    """
    N, rows, cols = C.shape
    D = np.zeros((N, rows, cols), dtype=np.float32)
    for i in tqdm(range(N), desc="Mark start"):
        valid_distances = C[i][C[i] != np.inf]
        threshold = np.percentile(valid_distances, 70)
        top_indices = np.argwhere((C[i] >= threshold) & (C[i] != np.inf))
        if len(top_indices) > 0:
            random_idx = np.random.choice(len(top_indices))
            j, k = top_indices[random_idx]
            D[i, j, k] = 1
    return D


def generate_BCD(A):
    B = create_tensor_B(A)
    C = create_tensor_C(A, B)
    D = create_tensor_D(C)
    C[C == np.inf] = 0  # чтобы лосс не убивался
    return np.stack([A, B, C, D], axis=-1)
