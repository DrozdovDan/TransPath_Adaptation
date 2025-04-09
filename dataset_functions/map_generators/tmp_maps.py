import numpy as np
import random
import os
from skimage.util import montage as montage2d

input_dir = "/home/silvarum/TransPath_Adaptation/results/32mp/"
all_tensor_files = [d for d in os.listdir(input_dir)]

def generate_tmp():
    """
    Генерирует карту 64x64, равномерно выбирая 4 типа карт и комбинируя их с помощью montage.
    
    Параметры:
    - data_dir: директория, содержащая .npy файлы с тензорами размера (100, 32, 32)
    
    Возвращает:
    - combined_map: тензор размера (64, 64), состоящий из 4 карт разных типов
    """
    # Выбираем 4 типа карт равномерно
    selected_types = random.sample(all_tensor_files, 4)
    
    # Загружаем только выбранные типы и выбираем случайные карты
    selected_maps = []
    for file_name in selected_types:
        file_path = input_dir + file_name
        tensor = np.load(file_path)
        selected_maps.append(tensor[random.randint(0, tensor.shape[0]-1)])
    
    # Преобразуем список в массив формы (4, 32, 32)
    selected_maps_array = np.array(selected_maps)
    
    # Используем montage для комбинирования карт в одно изображение 2x2
    combined_map = montage2d(selected_maps_array, grid_shape=(2, 2), padding_width=0)
    
    return combined_map