import numpy as np
import os
import glob
from PIL import Image
from skimage.filters import threshold_otsu
import random
import matplotlib.pyplot as plt


def generate_moving_street(input_dir = "/home/silvarum/planning-datasets/data/street/original/all", output_size=64):
    """
    Генерирует одну бинарную карту размера output_size x output_size.
    
    Функция выбирает случайное изображение из директории, вырезает из него фрагмент,
    затем преобразует его в бинарную карту с помощью порогового значения.
    
    Args:
        input_dir (str): путь к директории с исходными изображениями .png
        output_size (int): размер выходной матрицы (по умолчанию 64x64)
        
    Returns:
        np.ndarray: бинарная матрица размера output_size x output_size
    """
    # Ищем все подходящие файлы
    image_files = glob.glob(os.path.join(input_dir, "*_256.png"))
    
    if not image_files:
        raise FileNotFoundError(f"В директории {input_dir} не найдено файлов с шаблоном *_256.png")
    
    # Выбираем случайный файл
    random_file = random.choice(image_files)
    
    # Открываем и преобразуем в полутоновое изображение
    img = Image.open(random_file).convert("L").resize((256, 256))
    
    # Задаем размер вырезаемого фрагмента
    crop_size = 128
    
    # Генерируем случайную координату для левого верхнего угла фрагмента
    left = np.random.randint(0, 256 - crop_size)
    top = np.random.randint(0, 256 - crop_size)
    
    # Вырезаем фрагмент
    cropped = img.crop((left, top, left + crop_size, top + crop_size))
    
    # Изменяем размер фрагмента до требуемого
    resized = cropped.resize((output_size, output_size))
    
    # Преобразуем в массив numpy
    image_array = np.asarray(resized, dtype=np.float32)
    
    # Применяем пороговое значение для получения бинарной карты
    threshold = threshold_otsu(image_array)
    binary_map = np.zeros_like(image_array)
    binary_map[image_array > threshold] = 1.0
    
    return binary_map.astype(np.uint8)