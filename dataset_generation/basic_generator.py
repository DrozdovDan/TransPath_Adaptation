import numpy as np
from tqdm import tqdm

from ABCD.ABCD import generate_BCD
from map_generators.QR_maps import generate_QRs
from map_generators.band_maps import generate_bands
from map_generators.bug_trap_maps import generate_noisy_bug_traps
from map_generators.cave_maps import generate_caves
from map_generators.island_maps import generate_islands


label_to_generator = {
    "bands": generate_bands,
    "QRs": generate_QRs,
    "bug_traps": generate_noisy_bug_traps,
    "islands": generate_islands,
    "caves": generate_caves
    }


def basic_generator(label, N):
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
    generator = label_to_generator[label]
    return generate_BCD(np.array([generator() for _ in tqdm(range(N), desc=f"{label} are generating")]))
