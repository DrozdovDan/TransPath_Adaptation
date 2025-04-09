import random
import numpy as np

def generate_maze():
    grid = [[1 for _ in range(64)] for _ in range(64)]
    start = (0, 0)
    stack = [start]
    grid[0][0] = 0
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    
    def neighbors(cell):
        x, y = cell
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 64 and 0 <= ny < 64:
                result.append((nx, ny))
        return result
    
    while stack:
        current = stack[-1]
        unvisited = [n for n in neighbors(current) if grid[n[0]][n[1]] == 1]
        if unvisited:
            next_cell = random.choice(unvisited)
            # Прорезаем проход между current и next_cell
            mid = ((current[0] + next_cell[0]) // 2, (current[1] + next_cell[1]) // 2)
            grid[next_cell[0]][next_cell[1]] = 0
            grid[mid[0]][mid[1]] = 0
            stack.append(next_cell)
        else:
            stack.pop()
    return 1 - np.array(grid)

# print("\nКарта 21: Лабиринт с возвратами (DFS)")
# plt.imshow(generate_maze_dfs(), cmap='gray')
# plt.show()
