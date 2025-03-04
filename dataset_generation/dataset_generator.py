import numpy as np
from dataset_generation.advanced_generator import bad_generations_replacement_if_needed
from dataset_generation.basic_generator import label_to_generator, basic_generator


def generate_hard_map_by_label(label, N):
    """
    Функция пытается сгенерировать {N} карт вида {label}

    Параметры:
        generator: функция-генератор какого-то вида карт
        label: строка, в которой написан вид карт
        N: количество карт

    Возвращает:
        gen: (N, 64, 64, 4)-shaped ndarray. Целый кусок датасета,
        ассоциированный с какой-то конкретной картой
    """
    gen = basic_generator(label, N)
    bad_generations_replacement_if_needed(gen, label)
    return gen


def generate_dataset(N):
    res = np.empty((N, 64, 64, 4), dtype=np.float32)
    current_index = 0
    for label in label_to_generator.keys():
        if label in ["bands", "bug_traps", "QRs"]:
            res[current_index:current_index + N//4] = generate_hard_map_by_label(label, N // 4)
            current_index += N//4

        elif label in ["islands", "caves"]:
            res[current_index:current_index + N//8] = generate_hard_map_by_label(label, N // 8)
            current_index += N//8
        else:
            print("WTF?!")
    return res

def generate_dataset_by_label(label, N):
    res = np.empty((N, 64, 64, 4), dtype=np.float32)
    current_index = 0
    res[current_index:current_index + N] = generate_hard_map_by_label(label, N)
    current_index += N
    return res 
