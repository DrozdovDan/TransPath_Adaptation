import numpy as np
import heapq


def dijkstra_single_map(walkability_map, start):
    """
    Алгоритм Дейкстры для одной карты с 8-связной сеткой.

    Параметры:
        walkability_map (np.ndarray): 2D массив, задающий карту.
                                      0 - непроходимая клетка
                                      >0 - проходимая клетка (если нужны разные стоимости,
                                           можно хранить вес в каждой ячейке)
        start (tuple): Координаты старта (row, col).

    Возвращает:
        np.ndarray: Матрица расстояний от точки старта.
                    Если клетка недостижима, там будет значение np.inf.
    """
    # Размеры карты
    rows, cols = walkability_map.shape

    # Инициализируем матрицу расстояний
    distances = np.full((rows, cols), np.inf, dtype=float)
    distances[start[0], start[1]] = 0.0  # Расстояние до старта равно 0

    # Массив для отслеживания посещённых клеток (опционально)
    visited = np.zeros((rows, cols), dtype=bool)

    # Очередь с приоритетом для выбора следующей клетки с минимальным расстоянием
    # Храним кортеж (расстояние, (row, col))
    pq = [(0.0, start)]

    # Список направлений 8-связности
    cost_and_directions = [
        (1, -1,  0),  # вверх
        (1, 1,  0),  # вниз
        (1, 0, -1),  # влево
        (1, 0,  1),  # вправо
        (np.sqrt(2), -1, -1),  # вверх-влево
        (np.sqrt(2), -1,  1),  # вверх-вправо
        (np.sqrt(2), 1, -1),  # вниз-влево
        (np.sqrt(2), 1,  1),  # вниз-вправо
    ]

    # Пока очередь с приоритетом не опустеет
    while pq:
        current_dist, (r, c) = heapq.heappop(pq)

        # Если клетка уже посещена, пропускаем
        if visited[r, c]:
            continue
        visited[r, c] = True

        # Перебираем всех соседей
        for cost, dr, dc in cost_and_directions:
            nr, nc = r + dr, c + dc

            # Проверяем границы и проходимость
            if 0 <= nr < rows and 0 <= nc < cols and walkability_map[nr, nc] > 0:
                # Стоимость перехода в соседа = текущая + вес клетки-соседа
                new_dist = current_dist + cost
                # Если нашли более короткий путь, обновляем расстояние
                if new_dist < distances[nr, nc]:
                    distances[nr, nc] = new_dist
                    # Добавляем соседа в очередь с приоритетом
                    heapq.heappush(pq, (new_dist, (nr, nc)))

    return distances
