import numpy as np


def generate_pyramid():
    N = 64
    grid = [[0 for _ in range(N)] for _ in range(N)]
    top, bottom = 0, N - 1
    left, right = 0, N - 1
    direction = 0
    step = 2
    while top <= bottom and left <= right:
        if direction == 0:
            for j in range(left, right + 1):
                grid[top][j] = 1
            top += step
        elif direction == 1:
            for i in range(top, bottom + 1):
                grid[i][right] = 1
            right -= step
        elif direction == 2:
            for j in range(right, left - 1, -1):
                grid[bottom][j] = 1
            bottom -= step
        elif direction == 3:
            for i in range(bottom, top - 1, -1):
                grid[i][left] = 1
            left += step
        direction = (direction + 1) % 4
    return 1 - np.array(grid)


def generate_mask():
    arr1 = np.array([[i, i-1] for i in range(30, 0, -2)])
    arr2 = np.array([[i + 1, i] for i in range(32, 62, 2)])[::-1]
    arr3 = np.array([[i + 1 , 63 - i] for i in range(2, 32, 2)])
    arr4 = np.array([[62 - i, i] for i in range(0, 30, 2)])
    arr = np.stack([arr1, arr2, arr3, arr4], axis=1)

    remove_counts = np.random.choice([1, 2, 3], size=15, p=[0.25, 0.5, 0.25])
    random_indices = np.argsort(np.random.rand(15, 4), axis=1) 
    mask = random_indices >= remove_counts[:, None]  # (30, 4)
    arr[mask, :] = 0

    valid_mask = (arr != 0).any(axis=2)
    indices_to_zero = arr[valid_mask]
    return indices_to_zero


def generate_masked_pyramid():
    pyramid = generate_pyramid()
    mask = generate_mask()
    pyramid[mask[:, 0], mask[:, 1]] = 0
    return pyramid
