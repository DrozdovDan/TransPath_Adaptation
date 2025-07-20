import numpy as np
variety_threshold = 553 * 4    # np.sum(dijkstra_single_map(np.ones((11, 11)), (6, 6))) округленное вверх
hardness_threshold = 1.05


# TODO: объяснить что тут происходит
def reachability_variety(T):
    C = T[..., 2]
    return list(np.where(np.sum(C, axis=(1, 2), where=np.isfinite(C)) < variety_threshold)[0])


def octile_distance(p1, p2, diag_cost=np.sqrt(2), straight_cost=1):
    x1, y1 = p1
    x2, y2 = p2
    dx = np.abs(x2 - x1)
    dy = np.abs(y2 - y1)
    return diag_cost * np.minimum(dx, dy) + straight_cost * np.abs(dx - dy)


def find_digit_1(tensor):
    positions = [np.argwhere(tensor[k] == 1)[0] for k in range(tensor.shape[0])]
    return np.array(positions)


# TODO: объяснить что тут происходит
def reachability_hardness(T):
    B, C, D = T[..., 1], T[..., 2], T[..., 3]
    start = find_digit_1(D)
    end = find_digit_1(B)
    p1 = start[..., 0], start[..., 1]
    p2 = end[..., 0], end[..., 1]
    h = octile_distance(p1, p2)
    indices = []
    for i in range(C.shape[0]):
        if h[i] != 0 and not np.isnan(h[i]):  # эта проверка тут не просто так
            if (C[i, p1[0][i], p1[1][i]] / h[i]) < hardness_threshold:
                indices.append(i)
    return indices


def find_bad_gen_indices(T):
    """
    Проверяет, есть ли битые генерации во время генерации карт какого-то конкретного вида
    """
    bad_by_first_metric = reachability_variety(T)
    bad_by_second_metric = reachability_hardness(T)
    return np.array(list(set(bad_by_first_metric + bad_by_second_metric)))
