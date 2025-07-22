import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib.path import Path

# --- Общий костяк генерации фигур ---

def generate_centers(n, map_size=64):
    """
    Генерирует массив центров фигур в квадрате [0, map_size] x [0, map_size].
    Возвращает массив размера (n, 2).
    """
    centers = np.random.uniform(0, map_size, size=(n, 2))
    return centers

def assign_figure_types(n, type_ratios):
    """
    Назначает для каждого из n центров тип фигуры.
    type_ratios – словарь, где ключ – имя фигуры, значение – вес.
    Использует np.random.choice для выбора с соответствующими вероятностями.
    """
    types = np.array(list(type_ratios.keys()))
    weights = np.array(list(type_ratios.values()), dtype=float)
    probabilities = weights / weights.sum()
    assigned_types = np.random.choice(types, size=n, p=probabilities)
    return assigned_types

def sample_area(S_remaining, remaining_count, gamma_shape=5):
    """
    Сэмплирует площадь фигуры, используя гамма‑распределение с параметром формы gamma_shape.
    При этом математическое ожидание равно m = S_remaining/remaining_count.
    Если сэмплированная площадь превышает оставшуюся, повторяет сэмплирование.
    """
    m = S_remaining / remaining_count
    scale = m / gamma_shape  # m = gamma_shape * scale
    area = np.random.gamma(gamma_shape, scale)
    while area > S_remaining:
        area = np.random.gamma(gamma_shape, scale)
    return area

def sample_map(total_area, n_figures, type_ratios, figure_generator, map_size=64):
    """
    Генерирует набор фигур, располагаемых в области размера map_size x map_size.
    Распределяет общую площадь total_area между n_figures фигурами, выбирая для каждой центр, тип и площадь.
    Вызывает соответствующий генератор фигуры.
    Возвращает список сгенерированных фигур (их непрерывное представление).
    """
    centers = generate_centers(n_figures, map_size)
    assigned_types = assign_figure_types(n_figures, type_ratios)
    figures = []
    S_remaining = total_area

    for i in range(n_figures):
        remaining_count = n_figures - i
        if remaining_count > 1:
            area = sample_area(S_remaining, remaining_count)
        else:
            area = S_remaining
        fig_type = assigned_types[i]
        center = tuple(centers[i])
        figure = figure_generator(fig_type, center, area)
        figures.append(figure)
        S_remaining -= area
    return figures

# --- Генераторы фигур ---

def square_generator(fig_type, center, area):
    """
    Генератор для квадратиков.
    Вычисляет сторону по площади и формирует вершины квадрата, симметричного относительно центра.
    """
    side = math.sqrt(area)
    half_side = side / 2
    x, y = center
    vertices = [
        (x - half_side, y - half_side),
        (x + half_side, y - half_side),
        (x + half_side, y + half_side),
        (x - half_side, y + half_side)
    ]
    return {"type": fig_type, "center": center, "area": area, "vertices": vertices}

def circle_generator(fig_type, center, area):
    """
    Генератор для кружочков.
    Вычисляет радиус по формуле: area = π r²  =>  r = sqrt(area/π).
    """
    radius = math.sqrt(area / math.pi)
    return {"type": fig_type, "center": center, "area": area, "radius": radius}

def cross_generator(fig_type, center, area):
    """
    Генератор для крестиков с толщиной линий 2.
    Крестик состоит из двух пересекающихся прямоугольников толщиной 2.
    Если обозначить половину длины руки крестика как a, то суммарная площадь A = 8a – 4, откуда a = (area + 4)/8.
    Вычисляет 12 вершин внешнего контура для формирования сплошной непроходимой области.
    """
    t = 2  # Толщина линий крестика
    a = (area + 4) / 8.0
    half_t = t / 2.0
    cx, cy = center
    vertices = [
        (cx - a, cy - half_t),
        (cx - half_t, cy - half_t),
        (cx - half_t, cy - a),
        (cx + half_t, cy - a),
        (cx + half_t, cy - half_t),
        (cx + a, cy - half_t),
        (cx + a, cy + half_t),
        (cx + half_t, cy + half_t),
        (cx + half_t, cy + a),
        (cx - half_t, cy + a),
        (cx - half_t, cy + half_t),
        (cx - a, cy + half_t)
    ]
    return {"type": fig_type, "center": center, "area": area,
            "thickness": t, "arm_half_length": a, "vertices": vertices}

def figure_generator(fig_type, center, area):
    """
    Выбирает нужный генератор фигуры по типу.
    """
    if fig_type == "square":
        return square_generator(fig_type, center, area)
    elif fig_type == "circle":
        return circle_generator(fig_type, center, area)
    elif fig_type == "cross":
        return cross_generator(fig_type, center, area)
    else:
        raise ValueError(f"Неизвестный тип фигуры: {fig_type}")

# --- Растеризация фигур для дискретной карты ---

def rasterize_circle(center, radius, grid_size=64):
    """
    Возвращает булеву маску (grid_size x grid_size) для кружка.
    Клетка считается внутри, если расстояние от центра клетки (x+0.5, y+0.5) до центра кружка ≤ radius.
    """
    grid_x, grid_y = np.meshgrid(np.arange(grid_size) + 0.5, np.arange(grid_size) + 0.5)
    dist = np.sqrt((grid_x - center[0])**2 + (grid_y - center[1])**2)
    mask = dist <= radius
    return mask

def rasterize_polygon(vertices, grid_size=64):
    """
    Возвращает булеву маску (grid_size x grid_size) для многоугольника, заданного вершинами.
    """
    grid_x, grid_y = np.meshgrid(np.arange(grid_size) + 0.5, np.arange(grid_size) + 0.5)
    points = np.vstack((grid_x.ravel(), grid_y.ravel())).T
    path = Path(vertices)
    mask = path.contains_points(points)
    return mask.reshape((grid_size, grid_size))

def sample_discrete_map(figures, grid_size=64):
    """
    Генерирует дискретную карту (матрицу grid_size x grid_size),
    где 1 – проходимая клетка, 0 – клетка с препятствием.
    Если клетка попадает хотя бы в одну фигуру, её значение становится 0.
    """
    grid = np.ones((grid_size, grid_size), dtype=np.int8)
    for fig_item in figures:
        ftype = fig_item["type"]
        if ftype in ("square", "cross"):
            mask = rasterize_polygon(fig_item["vertices"], grid_size)
            grid[mask] = 0
        elif ftype == "circle":
            mask = rasterize_circle(fig_item["center"], fig_item["radius"], grid_size)
            grid[mask] = 0
    return grid

# --- Генерация случайных параметров ---

def sample_random_parameters():
    """
    Генерирует случайные параметры:
      - total_area ~ U(750, 1500)
      - n_figures – целое число из [10, 30]
      - type_ratios – случайные веса для типов "square", "circle", "cross"
    """
    total_area = np.random.uniform(750, 1500)
    n_figures = np.random.randint(10, 31)
    types = ["square", "circle", "cross"]
    weights = np.random.rand(3)
    type_ratios = dict(zip(types, weights))
    return total_area, n_figures, type_ratios

# --- Финальная функция генерации дискретной карты ---

def generate_figures(grid_size=128):
    """
    Финальная функция, которая не принимает входных параметров.
    Генерирует случайные параметры, создает набор фигур и возвращает дискретную карту (матрицу)
    размером grid_size x grid_size, где 1 – проходимая, 0 – препятствие.
    """
    total_area, n_figures, type_ratios = sample_random_parameters()
    figures = sample_map(total_area, n_figures, type_ratios, figure_generator, map_size=grid_size)
    discrete_map = sample_discrete_map(figures, grid_size=grid_size)
    # Возвращаем также параметры для отображения в заголовке (опционально)
    return discrete_map & np.random.choice([0, 1], size=(grid_size, grid_size), p=[0.15, 0.85])