import numpy as np

STEP      = 2            # толщина “витка” пирамиды

# ────────── 1. Пирамида ──────────
def generate_pyramid(grid_size: int = 128) -> np.ndarray:
    grid   = np.zeros((grid_size, grid_size), dtype=int)
    top, bottom = 0, grid_size - 1
    left, right = 0, grid_size - 1
    direction   = 0              # 0-вправо, 1-вниз, 2-влево, 3-вверх

    while top <= bottom and left <= right:
        if direction == 0:                 # верхняя сторона
            grid[top, left:right + 1] = 1
            top += STEP
        elif direction == 1:               # правая сторона
            grid[top:bottom + 1, right] = 1
            right -= STEP
        elif direction == 2:               # нижняя сторона
            grid[bottom, right:left - 1:-1] = 1
            bottom -= STEP
        else:                              # левая сторона
            grid[bottom:top - 1:-1, left] = 1
            left += STEP
        direction = (direction + 1) % 4

    return 1 - grid                        # инвертируем: 1 → “проход”, 0 → “стена”

# ────────── 2. Маска ──────────
def generate_mask(grid_size: int = 128) -> np.ndarray:
    """
    Формируем четыре набора координат вдоль «рукавов» пирамиды и
    случайно обнуляем по 1‒3 координаты в каждой строке (точно как в оригинале).
    """
    half      = grid_size // 2             # середина стороны
    segment_n = (half - 2) // 2            # длина каждого arr*  (15 при 64, 31 при 128)

    # Левая верхняя диагональ
    arr1 = np.array([[i, i - 1]                  for i in range(half - 2, 0, -2)])
    # Правая верхняя диагональ
    arr2 = np.array([[i + 1, i]                  for i in range(half, grid_size - 2, 2)])[::-1]
    # Правая нижняя диагональ
    arr3 = np.array([[i + 1, grid_size - 1 - i]  for i in range(2, half, 2)])
    # Левая нижняя диагональ
    arr4 = np.array([[grid_size - 2 - i, i]      for i in range(0, half - 2, 2)])

    arr  = np.stack([arr1, arr2, arr3, arr4], axis=1)  # shape: (segment_n, 4, 2)

    # Случайно “выбиваем” по 1-3 координаты из каждой четвёрки
    remove_counts  = np.random.choice([1, 2, 3], size=segment_n, p=[0.25, 0.5, 0.25])
    random_indices = np.argsort(np.random.rand(segment_n, 4), axis=1)
    mask           = random_indices >= remove_counts[:, None]
    arr[mask]      = 0

    # Оставляем только ненулевые координаты
    valid_rows = (arr != 0).any(axis=2)
    indices_to_zero = arr[valid_rows]

    return indices_to_zero.astype(int)

# ────────── 3. Итоговая функция ──────────
def generate_masked_pyramid(grid_size: int = 128) -> np.ndarray:
    pyramid = generate_pyramid(grid_size)
    mask    = generate_mask(grid_size)
    pyramid[mask[:, 0], mask[:, 1]] = 0
    return pyramid
