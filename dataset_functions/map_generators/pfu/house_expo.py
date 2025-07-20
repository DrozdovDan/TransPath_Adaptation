import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def json_to_square_matrix(json_data, grid=64, meter2pixel=100, border_pad=25, center=True):
    """
    Конвертирует JSON с вершинами в квадратную бинарную матрицу заданного размера.
    
    Параметры:
    - json_data: словарь с данными JSON
    - target_size: целевой размер стороны квадратной матрицы
    - meter2pixel: коэффициент преобразования метров в пиксели
    - border_pad: отступ от границ в пикселях
    - center: центрировать ли изображение в матрице
    """
    # Получаем вершины и масштабируем их
    verts = np.array(json_data["verts"]) * meter2pixel
    verts = verts.astype(int)
    
    # Используем bbox из JSON, если доступен
    if "bbox" in json_data:
        # Масштабируем минимальные и максимальные значения
        x_min = int(json_data["bbox"]["min"][0] * meter2pixel)
        y_min = int(json_data["bbox"]["min"][1] * meter2pixel)
        x_max = int(json_data["bbox"]["max"][0] * meter2pixel)
        y_max = int(json_data["bbox"]["max"][1] * meter2pixel)
    else:
        # Находим границы карты из вершин
        x_min, y_min = np.min(verts, axis=0)
        x_max, y_max = np.max(verts, axis=0)
    
    # Определяем размеры оригинальной матрицы
    orig_width = x_max - x_min + border_pad * 2
    orig_height = y_max - y_min + border_pad * 2
    
    # Смещаем вершины относительно минимальных координат и добавляем отступ
    verts[:, 0] = verts[:, 0] - x_min + border_pad
    verts[:, 1] = verts[:, 1] - y_min + border_pad
    
    # Определяем размер квадратной матрицы
    if max(orig_width, orig_height) < grid:
        # Если исходный размер меньше целевого, используем целевой размер
        square_size = grid
    else:
        # Используем ближайшую степень двойки, большую чем исходный размер
        square_size = grid
    
    # Масштабируем вершины к целевому размеру напрямую
    scale = square_size / max(orig_width, orig_height)
    verts = (verts * scale).astype(int)
    
    # Центрируем изображение в квадратной матрице, если требуется
    if center:
        # Размеры масштабированного изображения
        scaled_width = int(orig_width * scale)
        scaled_height = int(orig_height * scale)
        
        # Вычисляем смещение для центрирования
        offset_x = (square_size - scaled_width) // 2
        offset_y = (square_size - scaled_height) // 2
        
        # Смещаем вершины
        verts[:, 0] += offset_x
        verts[:, 1] += offset_y
    
    # Создаем квадратную матрицу
    matrix = np.ones((square_size, square_size), dtype=np.uint8)
    
    # Рисуем линии стен прямо на матрице целевого размера
    for i in range(len(verts)):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % len(verts)]
        
        # Проверяем корректность координат и рисуем линию
        if (0 <= x1 < square_size and 0 <= y1 < square_size and 
            0 <= x2 < square_size and 0 <= y2 < square_size):
            draw_line(matrix, x1, y1, x2, y2)
    
    # Используем Path для определения внутренних областей
    polygon_path = Path(verts)
    
    # Создаем сетку точек для проверки
    y_grid, x_grid = np.mgrid[:square_size, :square_size]
    points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    # Определяем, какие точки внутри многоугольника
    mask = polygon_path.contains_points(points)
    mask = mask.reshape(square_size, square_size)
    
    # Отмечаем внутренние области, сохраняя стены (0)
    inside_matrix = np.ones((square_size, square_size), dtype=np.uint8)
    inside_matrix[mask] = 1  # Внутренние точки
    inside_matrix[~mask] = 0  # Внешние точки
    
    # Совмещаем информацию о стенах и внутренних/внешних областях
    # Если точка была отмечена как стена (0), она остается стеной
    # Иначе используем информацию о внутренней/внешней области
    final_matrix = inside_matrix.copy()
    final_matrix[matrix == 0] = 0  # Сохраняем стены
    
    # Если требуется размер меньше созданной матрицы, выполняем уменьшение
    if square_size > grid:
        final_matrix = downsample_matrix(final_matrix, grid)
    
    return final_matrix

def draw_line(matrix, x0, y0, x1, y1):
    """
    Рисует линию между двумя точками на матрице (алгоритм Брезенхема).
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        # Проверяем границы
        if 0 <= y0 < matrix.shape[0] and 0 <= x0 < matrix.shape[1]:
            matrix[y0, x0] = 0  # Стена = 0
        
        if x0 == x1 and y0 == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

def downsample_matrix(matrix, grid):
    """
    Нерекурсивная функция для уменьшения размера матрицы до целевого размера.
    Сохраняет стены при уменьшении.
    """
    if matrix.shape[0] <= grid:
        return matrix
    
    # Вычисляем размер блока для уменьшения
    block_size = matrix.shape[0] // grid
    
    # Создаем матрицу целевого размера
    result = np.ones((grid, grid), dtype=np.uint8)
    
    # Для каждой ячейки результата проверяем соответствующий блок в исходной матрице
    for y in range(grid):
        y_start = y * block_size
        y_end = min((y + 1) * block_size, matrix.shape[0])
        
        for x in range(grid):
            x_start = x * block_size
            x_end = min((x + 1) * block_size, matrix.shape[1])
            
            # Вырезаем блок из исходной матрицы
            block = matrix[y_start:y_end, x_start:x_end]
            
            # Если в блоке есть стены, сохраняем стену
            if np.any(block == 0):
                result[y, x] = 0
    
    return result


def process_single_random_json(json_dir, grid=64, output_npy_file=None,):
    """
    Случайно выбирает и обрабатывает один JSON-файл из директории и создает матрицу numpy и PNG-изображение.
    
    Параметры:
    - json_dir: директория с JSON-файлами
    - output_npy_file: имя файла для сохранения матрицы numpy (если None, не сохраняется)
    - output_png_file: путь для сохранения PNG-изображения (если None, не сохраняется)
    - matrix_size: размер стороны квадратной матрицы
    
    Возвращает:
    - matrix: обработанная матрица
    - map_id: ID выбранного файла
    """
    # Получаем список всех JSON-файлов в директории
    all_json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    total_files = len(all_json_files)
    
    if total_files == 0:
        raise FileNotFoundError(f"В директории {json_dir} не найдено JSON-файлов")
    
    # print(f"Найдено {total_files} JSON-файлов в директории")
    
    # Выбираем один случайный файл
    random_file = random.choice(all_json_files)
    map_id = os.path.splitext(random_file)[0]  # Удаляем расширение .json
    
    # Путь к JSON-файлу
    json_path = os.path.join(json_dir, random_file)
    
    # Загружаем JSON
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Создаем матрицу
    matrix = json_to_square_matrix(
        json_data, 
        grid,
        meter2pixel=100,
        border_pad=25,
        center=True
    )
    
    # Если нужно, сохраняем матрицу в файл numpy
    if output_npy_file:
        np.save(output_npy_file, matrix)
        print(f"Матрица сохранена в {output_npy_file}, форма: {matrix.shape}")
    

    
    return matrix, map_id

def append_to_file(filename, text):
    """
    Добавляет текст в конец текстового файла.
    
    Параметры:
    - filename: имя файла или путь к файлу
    - text: текст для добавления
    """
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text)

def generate_house_expo(grid=128):
    A, map_id = process_single_random_json(json_dir="/home/silvarum/HouseExpo/HouseExpo/json", grid=grid)
    while (map_id == "17d28c7ece8ec16fd287295dd95aca85"): # len(ones_indices) == 0
         A, map_id = process_single_random_json(json_dir="/home/silvarum/HouseExpo/HouseExpo/json", grid=grid)
         append_to_file("/home/silvarum/TransPath_Adaptation/results/logs.txt", f"{map_id}\n")
    return A
