# map_generators/pfu/__init__.py

import inspect

from .maze               import generate_maze
from .Perlin             import generate_Perlin
from .recursive_division import generate_recursive_division
from .rotational_symmery import generate_rotational_symmetry
from .house_expo         import generate_house_expo
from .moving_street      import generate_moving_street
from .baldurs_gate       import generate_baldurs_gate
from .dcaffo             import generate_dcaffo
from .tmp                import generate_tmp
from .masked_pyramid     import generate_masked_pyramid

# Построим словарь label → функция автоматически
# Убираем префикс "generate_" и приводим к нижнему регистру
pfu = {
    func_name.replace("generate_", "").lower(): func
    for func_name, func in globals().items()
    if inspect.isfunction(func) and func_name.startswith("generate_")
}

__all__ = [
    "generate_maze",
    "generate_Perlin",
    "generate_recursive_division",
    "generate_rotational_symmetry",
    "generate_house_expo",
    "generate_moving_street",
    "generate_baldurs_gate",
    "generate_dcaffo",
    "generate_tmp",
    "generate_masked_pyramid",
    "pfu"
]
