import numpy as np
from tqdm import tqdm

from ABCD.ABCD import generate_BCD
from map_generators.pfu    import *
from map_generators.figures import *
from utils import find_bad_gen_indices


def basic_generator(dataset, label, N):
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
    generator = dataset[label]
    return generate_BCD(np.array([generator() for _ in tqdm(range(N), desc=f"{label} are generating")]))


def chAnge(label, num_of_bad_generations):
    """
    Пытается сгенерировать {num_of_bad_generations} корректных
     генераций (чтобы на уровне выше заменили некорректные на них)

    Параметры:
        generator: функция-генератор какого-то вида карт
        label: строка, в которой написан вид карт
        num_of_bad_generations: количество карт

    Возвращает:
        change: Тензор-замена. Содержит ровно {num_of_bad_generations}
        не битых генераций
    """

    flag = True
    i = 3
    while flag:
        if i > 10:
            print(f"Я не справился с тем, чтобы сгенерировать {label}")
            exit(52)
        change = basic_generator(label, i * num_of_bad_generations)
        bag_gen_indices = find_bad_gen_indices(change)
        # В таком случае не сможем заменить плохо сгенерированные изначально
        if len(bag_gen_indices) > num_of_bad_generations * (i - 1):
            print(f"Генерируя {label}, не получилось покрыть {num_of_bad_generations} плохих генераций хорошими, "
                  f"перегенировав {num_of_bad_generations * i} карт.")
            i += 1
            continue
        else:
            flag = False

    # хз работает нет
    if len(bag_gen_indices) != 0 and min(bag_gen_indices) < num_of_bad_generations:
        mask = np.ones(change.shape[0], dtype=bool)  # Создаем маску с True
        mask[bag_gen_indices] = False  # Убираем индексы из списка
        change = change[mask]
    return change[:num_of_bad_generations]


def bad_generations_replacement_if_needed(gen_to_fix, label):
    broken_gen = find_bad_gen_indices(gen_to_fix)
    num_of_bad_generations = len(broken_gen)
    if num_of_bad_generations != 0:
        replacement = chAnge(label, num_of_bad_generations)
        for i in range(num_of_bad_generations):
            index = broken_gen[i]
            gen_to_fix[index] = replacement[i]



def advanced_generator(dataset, label, N):
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
    gen = basic_generator(dataset, label, N)
    bad_generations_replacement_if_needed(gen, label)
    return gen


def generate_dataset(dataset, N):
    q = len(dataset)
    assert(N % q == 0), "quantity of topologies should divide quantity of maps"
    chunk_size = N // q
    parts = [advanced_generator(dataset, label, chunk_size) for label in dataset]
    return np.concatenate(parts, axis=0).astype(np.float32)



# def generate_dataset(dataset, N):
#     res = np.empty((N, 128, 128, 4), dtype=np.float32)
#     current_index = 0
#     q = len(dataset)
#     for label in dataset.keys():
#         res[current_index:current_index + N//q] = generate_hard_map_by_label(label, N // q)
#         current_index += N//q
#     return res

# def generate_dataset_by_label(label, N):
#     return generate_hard_map_by_label(label, N) 
