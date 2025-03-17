import random
import numpy as np


def create_trap(grid, start_row, start_col, width, length, removed_side):
    # Создание П-образной ловушки
    if removed_side == 'top':
        grid[start_row, start_col:start_col + width] = 0
        grid[start_row + length, start_col:start_col + width] = 0
        grid[start_row:start_row + length + 1, start_col] = 0
    elif removed_side == 'bottom':
        grid[start_row, start_col:start_col + width] = 0
        grid[start_row + length, start_col:start_col + width] = 0
        grid[start_row:start_row + length + 1, start_col + width] = 0
    elif removed_side == 'left':
        grid[start_row:start_row + length + 1, start_col] = 0
        grid[start_row:start_row + length + 1, start_col + width] = 0
        grid[start_row, start_col:start_col + width + 1] = 0
    elif removed_side == 'right':
        grid[start_row:start_row + length + 1, start_col] = 0
        grid[start_row:start_row + length + 1, start_col + width] = 0
        grid[start_row + length, start_col:start_col + width + 1] = 0


def generate_noisy_bug_traps(noise_level=0.2):
    # Инициализация основной матрицы 64x64
    grid = np.ones((64, 64), dtype=np.float32)

    # Ловушки живут внутри полос 8x64
    trap_zones = [(0, 8), (10, 18), (20, 28), (30, 38), (40, 48), (50, 58)]

    for start, end in trap_zones:
        num_traps = random.randint(6, 6)
        trap_width = 64 // (num_traps + 1)
        r = random.randint(0, 1)  # поправка, чтобы решетка клеток была как и чуть левее/правее чем равномерно
        for i in range(num_traps):
            # Вычисляем начальную позицию для текущей ловушки
            trap_col = (r + i) * trap_width + random.randint(0, trap_width - 8)

            width = random.randint(5, 7)
            length = random.randint(5, 7)
            trap_row = random.randint(start, end - length - 1)
            removed_side = random.choice(['top', 'bottom', 'left', 'right'])
            create_trap(grid, trap_row, trap_col, width, length, removed_side)

    # НУ-КА ДАЙТЕ ШУМА (только превращение проходимых клеток в непроходимые)
    noise_mask = np.random.random(grid.shape) < noise_level
    grid[noise_mask & (grid == 1)] = 0
    return grid

