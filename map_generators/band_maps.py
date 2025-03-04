import random
import numpy as np


def generate_unique_numbers_no_neighbors(n):
    all_numbers = np.arange(64)
    selected_numbers = []
    while len(selected_numbers) < n:
        choice = np.random.choice(all_numbers)
        selected_numbers.append(choice)
        all_numbers = all_numbers[
            (all_numbers != choice) & (all_numbers != choice - 1) & (all_numbers != choice + 1)
        ]
    return np.sort(selected_numbers)


def band_boundaries(num_bands):
    points = generate_unique_numbers_no_neighbors(num_bands)
    A = points[:-1]
    C = points[1:]
    # Генерация правого конца B_i для A_i в диапазоне [A_i, С_i - 1)
    B = np.array([np.random.randint(a, c-1) for a, c in zip(A, C)])
    return A, B


def cut_of_band_boundaries(num_bands):
    points = np.sort(np.random.choice(range(64), num_bands, replace=False))
    A = points[:-1]
    C = points[1:]
    B = np.array([np.random.randint(a, c) for a, c in zip(A, C)])
    # Создаем пары
    pairs = np.column_stack((A, B))
    np.random.shuffle(pairs)
    return pairs[:, 0], pairs[:, 1]


# Полосы
def generate_bands():
    num_bands = random.randint(5, 10)
    matrix = np.ones((64, 64), dtype=np.float32)

    A1, B1 = band_boundaries(num_bands + 1)  # Горизонтальные полосы
    A2, B2 = cut_of_band_boundaries(num_bands + 1)  # Вырезы

    # Вырезаем полосы
    for i in range(num_bands):
        matrix[A1[i]:B1[i] + 1] = 0
    # Возвращаем обратно вырезы из полос
    for j in range(num_bands):
        matrix[A1[j]:B1[j] + 1, A2[j]:B2[j] + 1] = 1
    return matrix
