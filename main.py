from dataset_functions.dataset_generation.dataset_generator import generate_dataset, generate_dataset_by_label
from dataset_functions.dataset_generation.basic_generator import label_to_generator
import numpy as np
import os

    # "pyramid": generate_masked_pyramid,
    # "maze": generate_maze,
    # "Perlin_noise": generate_Perlin_noise,
    # "random_lines": generate_random_lines, 
    # "recursive_division": generate_recursive_division,
    # "rotational_symmery_maps": generate_rotational_symmetry,
    # "house_expo_maps": generate_house_expo,
    # "moving_street_maps": generate_moving_street,
    # "baldurs_gate_maps": generate_baldurs_gate,
    # "dcaffo_maps": generate_dcaffo_maps





    # "maze": generate_maze,
    # "pyramid": generate_masked_pyramid,
    # "Perlin_noise": generate_Perlin_noise,
    # "recursive_division": generate_recursive_division,
    # "rotational_symmery_maps": generate_rotational_symmetry,
    # "dcaffo_maps": generate_dcaffo_maps,
    # "house_expo_maps": generate_house_expo,
    # "moving_street_maps": generate_moving_street,
    # "baldurs_gate_maps": generate_baldurs_gate,
    # "tmp_maps": generate_tmp

def create_directory_if_not_exists(directory_path):
    """
    Проверяет существование директории и создает ее, если она не существует.
    
    Args:
        directory_path (str): Путь к директории
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Директория '{directory_path}' создана")
    else:
        print(f"Директория '{directory_path}' уже существует")


import os
import numpy as np
import glob

def compress_npy_files(input_dir, output_dir):
    """
    Сжимает все .npy файлы из входной директории в выходную
    
    Args:
        input_dir: Путь к директории с .npy файлами
        output_dir: Путь к директории, куда будут сохранены сжатые файлы
    """
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Получаем список всех .npy файлов во входной директории
    npy_files = glob.glob(os.path.join(input_dir, "*.npy"))
    
    print(f"Найдено {len(npy_files)} .npy файлов для сжатия")
    
    # Обрабатываем каждый файл
    for file_path in npy_files:
        # Получаем имя файла без пути
        file_name = os.path.basename(file_path)
        # Создаем имя для выходного файла (меняем расширение с .npy на .npz)
        output_file = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.npz')
        
        try:
            # Загружаем numpy массив
            data = np.load(file_path)
            
            # Сохраняем с сжатием
            np.savez_compressed(output_file, data=data)
            print(f"Сжат файл: {file_name}")
            
        except Exception as e:
            print(f"Ошибка при обработке {file_path}: {e}")
    
    print(f"Сжатие завершено. Файлы сохранены в {output_dir}")

# Пример использования
if __name__ == "__main__":
    input_directory = "/home/silvarum/TransPath_Adaptation/results/train"  # Замените на путь к вашей папке с .npy файлами
    output_directory = "/home/silvarum/TransPath_Adaptation/results/zipped_train"  # Замените на путь, куда сохранить сжатые файлы
    
    compress_npy_files(input_directory, output_directory)

# if __name__ == '__main__':
#     print(list(label_to_generator.keys())[-4:])
#     create_directory_if_not_exists("/home/silvarum/TransPath_Adaptation/results/train")
#     for label in list(label_to_generator.keys())[-4:]:
#         print(label)
#         dr = f"/home/silvarum/TransPath_Adaptation/results/train/{label}"
#         data = generate_dataset_by_label(label, 16000)
#         np.save(dr, data)
    # plt.imshow(data[43, :, :, 0], cmap='gray')
    # plt.show()