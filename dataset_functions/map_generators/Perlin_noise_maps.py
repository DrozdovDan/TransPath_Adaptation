import random
import numpy as np

def generate_Perlin_noise():
    grid = [[random.randint(0, 1) for _ in range(64)] for _ in range(64)]

    # Применяем размытие: заменяем каждую ячейку на среднее её соседей
    for _ in range(2):  # Два прохода сглаживания
        new_grid = [[0 for _ in range(64)] for _ in range(64)]
        for i in range(64):
            for j in range(64):
                neighbors = [
                    grid[x][y] for x in range(max(0, i - 1), min(64, i + 2))
                               for y in range(max(0, j - 1), min(64, j + 2))
                ]
                new_grid[i][j] = 1 if sum(neighbors) > 4 else 0  # Усредняем
        grid = new_grid
    
    return 1 - np.array(grid)

# print("\nКарта 30: Случайная текстура (Perlin Noise-подобная)")
# plt.imshow(generate_random_texture(), cmap='gray')
# plt.show()
