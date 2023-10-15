import importlib
import sys


def reload(m: str):
    for name, module in list(sys.modules.items()):
        if name.startswith(m + '.'):
            importlib.reload(module)
