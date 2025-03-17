import random
import numpy as np


def step_quality(x1, y1, x2, y2, dx, dy):
    """Проверяет, приближает ли нас конкретный шаг к достижению цели"""
    if abs((x1 + dx) - x2) + abs((y1 + dy) - y2) < abs(x1 - x2) + abs(y1 - y2):
        return 'good'  # Хороший шаг (приближает к цели)
    else:
        return 'bad'  # Плохой шаг (отдаляет от цели)


def connect_points_and_mark_path(start, end, grid):
    """Соединяет точки между островами"""
    x1, y1 = start
    x2, y2 = end
    path = []

    # Обновляем стартовую точку
    grid[y1, x1] = 1
    path.append((x1, y1))
    possible_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while x1 != x2 or y1 != y2:
        # В большинстве случаев выбираем случайное отклонение
        action = random.choices(
            ['vertical', 'horizontal', 'random', 'bad'],  # Возможные действия
            weights=[0.25, 0.25, 0, 0.1],  # 50% шанс на случайное отклонение, 10% на плохой шаг
            k=1
        )[0]

        if action == 'vertical' and y1 != y2:
            if y1 < y2:
                y1 += 1
            elif y1 > y2:
                y1 -= 1
        elif action == 'horizontal' and x1 != x2:
            if x1 < x2:
                x1 += 1
            elif x1 > x2:
                x1 -= 1
        elif action == 'random':
            # Случайное отклонение
            dx, dy = random.choice(possible_steps)  # Случайный шаг по горизонтали или вертикали
            x1 += dx
            y1 += dy
        elif action == 'bad':
            # Плохой шаг, отдаляет от цели
            # Сначала выбираем случайный шаг, который отдаляет нас от цели
            random_steps = random.randint(1, 3)  # Сколько шагов в плохом направлении
            for _ in range(random_steps):
                dx, dy = random.choice(possible_steps)  # Случайный шаг по горизонтали или вертикали
                while step_quality(x1, y1, x2, y2, dx, dy) == 'good':
                    # Если шаг приближает нас к цели, пробуем выбрать другой
                    dx, dy = random.choice(possible_steps)
                # Делает плохой шаг
                x1 += dx
                y1 += dy
                x1 = np.clip(x1, 0, grid.shape[1] - 1)
                y1 = np.clip(y1, 0, grid.shape[0] - 1)
                grid[y1, x1] = 1

        # Проверяем, чтобы мы не вышли за пределы карты
        x1 = np.clip(x1, 0, grid.shape[1] - 1)
        y1 = np.clip(y1, 0, grid.shape[0] - 1)

        # Обновляем карту
        grid[y1, x1] = 1
        path.append((x1, y1))

        # Если мы достигли цели, можем закончить путь
        if x1 == x2 and y1 == y2:
            break

    return grid


def transitive_closure(matrix):
    """
    Преобразует матрицу достижимости в транзитивно замкнутую матрицу.
    Если есть проход с острова {i} на остров {j} через промежуточные острова, то
    мы обновляем матрицу так, что {matrix[i, j]} становится 1.
    """
    n = matrix.shape[0]
    # Алгоритм Флойда
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # Если есть путь через промежуточный остров k, то добраться можно
                if matrix[i, k] == 1 and matrix[k, j] == 1:
                    matrix[i, j] = 1
    return matrix


def building_bridge(islands, map_of_islands):
    island_connectivity = np.eye(len(islands), dtype=int)  # острова соединены сами собой
    while not np.all(island_connectivity == np.ones((len(islands), len(islands)), dtype=int)):
        # Выбрали два острова, которые еще не соединяли
        zero_indices = np.column_stack(np.where(island_connectivity == 0))
        first_oval, second_oval = random.choice(zero_indices)

        # Пытаемся соединить
        first_point = islands[first_oval][1], islands[first_oval][2]
        second_point = islands[second_oval][1], islands[second_oval][2]

        # Рисуем мостик между этими точками
        map_of_islands = connect_points_and_mark_path(first_point, second_point, map_of_islands)
        # Соединили
        island_connectivity[first_oval, second_oval] = 1
        # Теперь каждый из островов связан с соседями того, с кем мы его связали
        island_connectivity = transitive_closure(island_connectivity)


# Функция сглаживания карты
def random_walker(map_of_islands):
    deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for _ in range(8):
        walls_sum = np.zeros_like(map_of_islands, dtype=np.float32)
        for dx, dy in deltas:
            shifted = np.roll(map_of_islands, shift=dy, axis=0)
            shifted = np.roll(shifted, shift=dx, axis=1)
            walls_sum += (shifted == 0)
        window_sum = len(deltas)
        condition = (walls_sum / window_sum) < 0.5
        map_of_islands[condition] = 1
    return map_of_islands


#TODO: правда ли что 4 острова генерируются лучше чем 3? Можно ли генерировать больше адекватно?
def generate_islands(size=64, min_ovals=4, max_ovals=4):
    """
    Создает матрицу с овалами, равномерно распределенными по карте
    Соединяет овалы ломанными специального вида
    Возможно, сглаживает (см. {smooth_mode})
    """
    # Все клетки непроходимы
    map_of_islands = np.zeros((size, size), dtype=int)

    # Случайное количество овалов
    num_islands = random.randint(min_ovals, max_ovals)

    parts_of_map = [
        (0, 0, size // 2, size // 2),  # верхний левый квадрант
        (size // 2, 0, size - 1, size // 2),  # верхний правый квадрант
        (0, size // 2, size // 2, size - 1),  # нижний левый квадрант
        (size // 2, size // 2, size - 1, size - 1)  # нижний правый квадрант
    ]

    random.shuffle(parts_of_map)

    islands = []
    for part in parts_of_map[:num_islands]:
        flag = True
        while flag:
            x0, y0, x1, y1 = part

            # Случайные параметры овала в конкретной зоне
            center_x = random.randint(x0, x1)
            center_y = random.randint(y0, y1)

            # Размеры овала пропорционально размеру зоны
            zone_width = x1 - x0
            zone_height = y1 - y0

            radius_x = random.randint(zone_width // 8, zone_width // 4)
            radius_y = random.randint(zone_height // 8, zone_height // 4)

            # Создаем координатную сетку
            y, x = np.ogrid[:size, :size]

            # Уравнение овала
            oval_mask = ((x - center_x) / radius_x) ** 2 + ((y - center_y) / radius_y) ** 2 <= 1
            if np.sum(oval_mask) <= 1:
                continue

            islands.append([oval_mask, center_x, center_y])
            map_of_islands[oval_mask] = 1

            flag = False

    # Возводим мосты
    building_bridge(islands, map_of_islands)
    return map_of_islands
