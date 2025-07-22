import random
import numpy as np

def generate_maze(grid_size=128):
    # Создаём сетку, заполняем её стенами (1)
    grid = np.ones((grid_size, grid_size), dtype=int)
    start = (0, 0)
    stack = [start]
    grid[0, 0] = 0
    
    # Направления: верх, вниз, влево, вправо
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    # Функция для получения соседей
    def neighbors(cell):
        x, y = cell
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                result.append((nx, ny))
        return result
    
    # Множество посещённых клеток для быстрого доступа
    visited = np.zeros((grid_size, grid_size), dtype=bool)
    visited[0, 0] = True
    
    while stack:
        current = stack[-1]
        unvisited = [n for n in neighbors(current) if not visited[n[0], n[1]]]
        if unvisited:
            next_cell = random.choice(unvisited)
            # Прорезаем проход между current и next_cell
            mid = ((current[0] + next_cell[0]) // 2, (current[1] + next_cell[1]) // 2)
            grid[next_cell[0], next_cell[1]] = 0
            grid[mid[0], mid[1]] = 0
            visited[next_cell[0], next_cell[1]] = True
            stack.append(next_cell)
        else:
            stack.pop()

    return 1 - grid  # Инвертируем: 0 - проходим, 1 - стена


