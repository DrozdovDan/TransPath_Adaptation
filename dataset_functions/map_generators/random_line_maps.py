import random, math
import numpy as np

def generate_random_lines():
    grid = [[0 for _ in range(64)] for _ in range(64)]
    
    def draw_line(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < 64 and 0 <= y0 < 64:
                grid[y0][x0] = 1
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    # Рисуем, например, 50 случайных линий
    for _ in range(50):
        x0 = random.randint(0, 63)
        y0 = random.randint(0, 63)
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(5, 20)
        x1 = x0 + int(round(math.cos(angle) * length))
        y1 = y0 + int(round(math.sin(angle) * length))
        draw_line(x0, y0, x1, y1)
    
    return 1 - np.array(grid)

# print("\nКарта 23: Случайные линии")
# plt.imshow(generate_random_lines_map(), cmap='gray')
# plt.show()
