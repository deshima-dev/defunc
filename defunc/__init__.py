__version__ = '2018.10.26'


def tidy_namespace(module):
    for name in dir(module):
        if name.startswith('_'):
            continue

        if name in module.__all__:
            continue

        delattr(module, name)


from . import utils
from .funcs import *
tidy_namespace(utils)
tidy_namespace(funcs)