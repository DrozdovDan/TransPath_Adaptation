import numpy as np

def generate_rotational_symmetry(size=128):
    half = size // 2
    # Генерируем случайный узор для верхнего левого квадранта
    quadrant = np.random.choice([0, 1], (half, half))
    
    # Создаем пустую решетку
    grid = np.zeros((size, size), dtype=int)

    # Заполняем квадранты с использованием срезов
    grid[:half, :half] = quadrant  # Верхний левый квадрант
    grid[:half, half:] = np.fliplr(quadrant)  # Верхний правый квадрант (зеркально по горизонтали)
    grid[half:, :half] = np.flipud(quadrant)  # Нижний левый квадрант (зеркально по вертикали)
    grid[half:, half:] = np.flipud(np.fliplr(quadrant))  # Нижний правый квадрант (зеркально по обеим осям)

    return 1 - grid  # Инвертируем результат

