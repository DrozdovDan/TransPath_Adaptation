import random
import numpy as np

def generate_rotational_symmetry():
    size = 64
    half = size // 2
    # Генерируем случайный узор для верхнего левого квадранта
    quadrant = [[random.choice([0,1]) for _ in range(half)] for _ in range(half)]
    grid = [[0 for _ in range(size)] for _ in range(size)]
    # Заполняем верхний левый квадрант
    for i in range(half):
        for j in range(half):
            grid[i][j] = quadrant[i][j]
    # Зеркальное отражение по горизонтали – верхний правый квадрант
    for i in range(half):
        for j in range(half):
            grid[i][size-1-j] = quadrant[i][j]
    # Зеркальное отражение по вертикали – нижний левый квадрант
    for i in range(half):
        for j in range(half):
            grid[size-1-i][j] = quadrant[i][j]
    # Нижний правый квадрант – зеркальное отражение по обеим осям
    for i in range(half):
        for j in range(half):
            grid[size-1-i][size-1-j] = quadrant[i][j]
    return 1 - np.array(grid)

# print("\nКарта 60: 90° Rotational Symmetry")
# plt.imshow(generate_rotational_symmetry(), cmap='gray')
# plt.show()
