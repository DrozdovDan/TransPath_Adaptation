import random
import numpy as np

def generate_recursive_division(grid_size=128):
    # Инициализируем сетку из нулей (проход)
    grid = np.zeros((grid_size, grid_size), dtype=int)
    
    # Используем NumPy для случайного выбора значения 1 (стена) или 0 (проход)
    def divide(x, y, w, h, orientation):
        if w < 4 or h < 4:
            return
        if orientation == 'H':
            # Горизонтальное деление
            split = random.randint(y+1, y+h-2)
            grid[split, x:x+w] = np.random.choice([0, 1], size=w, p=[0.07, 0.93])
            # Рекурсивно делим
            divide(x, y, w, split-y, 'V')
            divide(x, split+1, w, y+h-split-1, 'V')
        else:  # Вертикальное деление
            split = random.randint(x+1, x+w-2)
            grid[y:y+h, split] = np.random.choice([0, 1], size=h, p=[0.07, 0.93])
            # Рекурсивно делим
            divide(x, y, split-x, h, 'H')
            divide(split+1, y, x+w-split-1, h, 'H')

    divide(0, 0, grid_size, grid_size, 'H')
    
    return 1 - grid  # Инвертируем: 0 - проходим, 1 - стена
