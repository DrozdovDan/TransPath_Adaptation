import math
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_hex_centers(width, height, cell_size):
    """
    Генерирует список центров гексагональных ячеек (x, y),
    равномерно покрывающих область [0, width] x [0, height].
    
    :param width:     Ширина области
    :param height:    Высота области
    :param cell_size: Расстояние между центрами ячеек по горизонтали
    :return:          Список (cx, cy) координат центров
    """
    centers = []
    # Шаг по вертикали в гекс-сетке (соседние ряды смещены)
    vertical_spacing = math.sqrt(3) * cell_size / 2.0
    
    # Примерная оценка, сколько рядов (по вертикали)
    num_rows = int(math.ceil(height / vertical_spacing)) + 1
    # Примерная оценка, сколько столбцов (по горизонтали)
    num_cols = int(math.ceil(width  / cell_size)) + 1
    
    for row in range(num_rows):
        # Смещение по вертикали
        cy = row * vertical_spacing
        
        # Для нечётных рядов смещаем центры по горизонтали на половину cell_size
        row_offset = 0.0
        if row % 2 == 1:
            row_offset = cell_size / 2.0
        
        for col in range(num_cols):
            cx = col * cell_size + row_offset
            # Проверяем, что (cx, cy) не выходит за границы
            if cx <= width and cy <= height:
                centers.append((cx, cy))
    
    return centers

def generate_hex_noise(width, height, cell_size=20, seed=None):
    """
    Генерирует 2D-карту (numpy-массив размера height x width) 
    гексагонального шума. Каждый пиксель привязывается к ближайшему 
    центру гекс-сетки, у которого есть случайное значение.
    
    :param width:     Ширина итогового массива
    :param height:    Высота итогового массива
    :param cell_size: Размер "гекс ячейки" по горизонтали
    :param seed:      Зерно для генератора случайных чисел (опционально)
    :return:          numpy-массив float, [0..1] размером (height, width)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # 1) Генерируем центры гексагональной сетки
    centers = generate_hex_centers(width, height, cell_size)
    
    # 2) Для каждого центра сохраним случайное значение от 0 до 1
    #    Можно сделать массив или словарь; здесь - просто список
    center_values = [random.random() for _ in centers]
    
    # Удобно хранить их вместе
    # [(cx, cy, val), ...]
    centers_data = list(zip(
        [c[0] for c in centers],
        [c[1] for c in centers],
        center_values
    ))
    
    # 3) Создаём пустую карту
    result = np.zeros((height, width), dtype=np.float32)
    
    # 4) Заполняем карту пиксель за пикселем
    #    - Находим ближайший центр (cx, cy)
    #    - Берём значение, ассоциированное с ним
    #
    #  В простейшем случае мы могли бы искать среди всех центров,
    #  но это O(N*M) где N=width*height, M=количество центров.
    #  Для больших размеров лучше делать пространственные структуры (квантование, grid lookup).
    #  Но для наглядности оставим простой подход или ограничим поиск локальными центрами.
    
    # Чтобы не сравнивать с *всеми* точками, сузим поиск к "окрестным" центрам:
    # - радиус поиска по x: ~ cell_size
    # - радиус поиска по y: ~ sqrt(3)*cell_size
    # Это существенно уменьшит количество проверок.
    
    # Сформируем дополнительную структуру для быстрого поиска:
    # Разделим нашу сцену на "ячейки" в том же шаге, что и hex:
    grid_w = int(math.ceil(width  / cell_size))
    grid_h = int(math.ceil(height / (math.sqrt(3)*cell_size/2.0)))
    
    # "grid_map[row][col]" будет содержать список индексов центров, попавших в эту "крупную ячейку"
    grid_map = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    
    def get_grid_coord(cx, cy):
        # Приблизительно переводим (cx, cy) в (row, col) для grid_map
        row = int(cy // (math.sqrt(3)*cell_size/2.0))
        col = int(cx // cell_size)
        return row, col
    
    # Заполняем grid_map индексами центров
    for idx, (cx, cy, val) in enumerate(centers_data):
        row, col = get_grid_coord(cx, cy)
        if 0 <= row < grid_h and 0 <= col < grid_w:
            grid_map[row][col].append(idx)
    
    # Функция для поиска ближайшего центра в окрестности
    def find_nearest_center(px, py):
        # Находим, в какую "крупную ячейку" попадает пиксель
        row, col = get_grid_coord(px, py)
        best_idx = None
        best_dist_sq = 1e10
        
        # Смотрим не только в текущей ячейке, но и в соседних
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                rr = row + dr
                cc = col + dc
                if 0 <= rr < grid_h and 0 <= cc < grid_w:
                    for idx_center in grid_map[rr][cc]:
                        cx, cy, val = centers_data[idx_center]
                        dx = px - cx
                        dy = py - cy
                        dist_sq = dx*dx + dy*dy
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            best_idx = idx_center
        return best_idx
    
    # Заполняем картинку
    for y in range(height):
        for x in range(width):
            idx_center = find_nearest_center(x, y)
            _, _, cval = centers_data[idx_center]
            result[y, x] = cval
    
    return result

def threshold_image(image, threshold=0.5):
    """
    Бинаризует (чёрно-белое) изображение:
    - Если значение >= threshold -> 1.0 (белое)
    - Иначе -> 0.0 (чёрное)
    """
    return (image >= threshold).astype(np.float32)


if __name__ == "__main__":
    # Параметры
    WIDTH, HEIGHT = 64, 64
    CELL_SIZE = 5     # Расстояние между центрами по горизонтали
    SEED = 42
    THRESHOLD = 0.45     # Порог для бинаризации (необязательно применять)

    # Генерируем гексагональный шум
    hex_noise_map = generate_hex_noise(WIDTH, HEIGHT, cell_size=CELL_SIZE)

    # Применяем бинаризацию для более «дискретного» вида
    bw_map = threshold_image(hex_noise_map, THRESHOLD)

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(hex_noise_map, cmap='gray', origin='upper')
    ax1.set_title("Hex Noise (grayscale)")
    ax1.axis('off')

    ax2.imshow(bw_map, cmap='gray', origin='upper')
    ax2.set_title(f"Hex Noise (threshold = {THRESHOLD})")
    ax2.axis('off')

    plt.show()
