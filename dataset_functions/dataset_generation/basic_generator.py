import numpy as np
from tqdm import tqdm

from dataset_functions.ABCD.ABCD import generate_BCD
from dataset_functions.map_generators.QR_maps import generate_QRs
from dataset_functions.map_generators.band_maps import generate_bands
from dataset_functions.map_generators.bug_trap_maps import generate_noisy_bug_traps
from dataset_functions.map_generators.cave_maps import generate_caves
from dataset_functions.map_generators.island_maps import generate_islands


from dataset_functions.map_generators.masked_pyramid_maps import generate_masked_pyramid
from dataset_functions.map_generators.maze_maps import generate_maze
from dataset_functions.map_generators.Perlin_noise_maps import generate_Perlin_noise
from dataset_functions.map_generators.random_line_maps import generate_random_lines
from dataset_functions.map_generators.recursive_division_maps import generate_recursive_division
from dataset_functions.map_generators.rotational_symmery_maps import generate_rotational_symmetry
from dataset_functions.map_generators.house_expo_maps import generate_house_expo
from dataset_functions.map_generators.moving_street_maps import generate_moving_street
from dataset_functions.map_generators.baldurs_gate_maps import generate_baldurs_gate
from dataset_functions.map_generators.dcaffo_maps import generate_dcaffo_maps
from dataset_functions.map_generators.tmp_maps import generate_tmp
from dataset_functions.map_generators.dumbQR_maps import generate_dumbQRs
from dataset_functions.map_generators.figures import generate_figures

label_to_generator = {
    ## AlekSet:
    # "bands": generate_bands,
    # "QRs": generate_QRs,
    # "bug_traps": generate_noisy_bug_traps,
    # "islands": generate_islands,
    # "caves": generate_caves,
    ## TestSet:
    # "maze": generate_maze,
    # "pyramid": generate_masked_pyramid,
    # "Perlin_noise": generate_Perlin_noise,
    # "recursive_division": generate_recursive_division,
    # "rotational_symmery_maps": generate_rotational_symmetry,
    # "dcaffo_maps": generate_dcaffo_maps,
    # "house_expo_maps": generate_house_expo,
    # "moving_street_maps": generate_moving_street,
    # "baldurs_gate_maps": generate_baldurs_gate,
    # "tmp_maps": generate_tmp,
    ## dumbTrainSet:
    # "dumbQRs": generate_dumbQRs,
    ## figures:
    "figures":  generate_figures

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
