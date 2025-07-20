import numpy as np

def generate_Perlin(grid_size: int = 128, passes: int = 2) -> np.ndarray:
    """
    Быстрое бинарное «перлин-шумоподобное» поле с двумерным сглаживанием.
    - grid_size : размер решётки (число, кратное 2)
    - passes    : сколько раз применять сглаживание
    Возвращает массив 0/1 той же формы.
    """
    # Начальная случайная бинарная решётка
    grid = np.random.randint(0, 2, (grid_size, grid_size), dtype=np.uint8)

    # 9-точечное окно (сама точка + 8 соседей)
    shifts = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1), ( 0, 0), ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]

    for _ in range(passes):
        # Суммируем все сдвинутые версии решётки
        smoothed = sum(np.roll(np.roll(grid, dx, 0), dy, 1) for dx, dy in shifts)
        # Порог: больше половины окон → 1, иначе 0
        grid = (smoothed > 4).astype(np.uint8)

    return 1 - grid
