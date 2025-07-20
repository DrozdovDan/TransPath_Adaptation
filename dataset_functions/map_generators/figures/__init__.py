# map_generators/pfu/__init__.py
import inspect
from .uniform              import generate_uniform
from .beta                 import generate_beta
from .figures              import generate_figures


__all__ = [
    "generate_uniform",
    "generate_beta",
    "generate_figures"
]

figures = {
    func_name.replace("generate_", "").lower(): func
    for func_name, func in globals().items()
    if inspect.isfunction(func) and func_name.startswith("generate_")
}
