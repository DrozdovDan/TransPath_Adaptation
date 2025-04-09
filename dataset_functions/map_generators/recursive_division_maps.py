import random
import numpy as np

def generate_recursive_division():
    grid = [[0 for _ in range(64)] for _ in range(64)]
    
    def divide(x, y, w, h, orientation):
        if w < 4 or h < 4:
            return
        if orientation == 'H':
            split = random.randint(y+1, y+h-2)
            for j in range(x, x+w):
                grid[split][j] = np.random.choice([0, 1], p = [0.2, 0.8])
            divide(x, y, w, split-y, 'V')
            divide(x, split+1, w, y+h-split-1, 'V')
        else:  # вертикальное деление
            split = random.randint(x+1, x+w-2)
            for i in range(y, y+h):

                grid[i][split] =  np.random.choice([0, 1], p = [0.2, 0.8])
            divide(x, y, split-x, h, 'H')
            divide(split+1, y, x+w-split-1, h, 'H')

    divide(0, 0, 64, 64, 'H')
    return 1 - np.array(grid)

# print("\nКарта 15: Рекурсивное деление")
# plt.imshow(generate_recursive_division_map(), cmap='gray')
# plt.show()