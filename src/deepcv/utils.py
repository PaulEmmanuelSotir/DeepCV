#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Utils module - deepcv.utils.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""

import os
import re
import imp
import sys
import time
import math
import types
import random
import inspect
import logging
import builtins
import threading
import importlib
from pathlib import Path
import importlib.machinery
from types import SimpleNamespace
from functools import singledispatch, partial
from typing import Union, Iterable, Optional, Dict, Any, List, Tuple, Sequence, Callable, Type

import numpy as np
from tqdm import tqdm

import torch
from kedro.io import DataCatalog


__all__ = ['Number', 'setup_cudnn', 'set_seeds', 'set_each_seeds', 'progess_bar', 'get_device', 'merge_dicts',
           'periodic_timer', 'cd', 'ask', 'human_readable_size', 'is_roughtly_constant', 'yolo', 'get_by_identifier', 'get_str_repr',
           'source_dir', 'try_import', 'import_pickle', 'import_and_reload', 'import_third_party', 'import_tests']
__author__ = 'Paul-Emmanuel Sotir'

Number = Union[builtins.int, builtins.float, builtins.bool]


def setup_cudnn(deterministic: bool = False, use_gpu: bool = torch.cuda.is_available()) -> types.ModuleType:
    if use_gpu:
        torch.backends.cudnn.deterministic = deterministic  # Makes training procedure reproducible (may have small performance impact)
        torch.backends.cudnn.benchmark = use_gpu and not torch.backends.cudnn.deterministic
        torch.backends.cudnn.fastest = use_gpu  # Disable this if memory issues
        return torch.backends.cudnn


@singledispatch
def set_seeds(all_seeds: int = 345349):
    set_each_seeds(torch_seed=all_seeds, cuda_seed=all_seeds, np_seed=all_seeds, python_seed=all_seeds)


@set_seeds.register(int)
def set_each_seeds(torch_seed: int = None, cuda_seed: int = None, np_seed: int = None, python_seed: int = None):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    if cuda_seed is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cuda_seed)
    if np_seed is not None:
        np.random.seed(np_seed)
    if python_seed is not None:
        random.seed(python_seed)


def progess_bar(iterable: Iterable, desc: str, disable: bool = False):
    return tqdm(iterable, unit='batch', ncols=180, desc=desc, ascii=False, position=0, disable=disable,
                bar_format='{desc} {percentage:3.0f}%|'
                '{bar}'
                '| {n_fmt}/{total_fmt} [Elapsed={elapsed}, Remaining={remaining}, Speed={rate_fmt}{postfix}]')


# def start_tensorboard(port=8889, forward=False):
#     raise NotImplementedError  # TODO: implementation


# def stop_tensorboard(port=8889):
#     raise NotImplementedError  # TODO: implementation


def get_device(devid: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda' + (f':{devid}' if devid else ''))
    return torch.device('cpu')


def merge_dicts(*dicts: Iterable[Dict[str, Any]]):
    """ Utils function used to merge given dictionnaries """
    merged = {}
    for p in dicts:
        merged.update(p)
    return merged


class periodic_timer:
    def __init__(self, func, period=1., args=[], kwargs={}):
        self.func = func
        self.period = period
        self._args = args
        self._kwargs = kwargs

    def stop(self):
        if self._timer is not None:
            self._timer.cancel()

    def start(self, count=-1):
        if count != 0:
            start = time.time()
            self.func(*self._args, **self._kwargs)
            delta_t = time.time() - start
            self._timer = threading.Timer(max(0, self.period - delta_t), self.start, args=[count - 1])
            self._timer.start()


class cd:
    """Context manager for changing the current working directory from https://stackoverflow.com/a/13197763/5323273"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


def ask(prompt: str, choices: List = ['N', 'Y'], ask_indexes: bool = False):
    prompt += ' (Choices: ' + ('; '.join([f'{i}: "{str(e)}""' for i, e in enumerate(choices)]) if ask_indexes else '; '.join(map(str, choices)))
    choice = input(prompt)

    if ask_indexes:
        while not choice.isdigit() or int(choice) not in range(len(choices)):
            choice = input(prompt)
        return int(choice), choices[int(choice)]
    else:
        while(choice not in choices):
            choice = input(prompt)
        return choices.index(choice), choice


def human_readable_size(size_bytes: int, format_to_str: bool = True) -> Union[str, Tuple[float, str]]:
    size_units = ('B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB')
    if size_bytes == 0:
        return 0., size_units[0]

    # Handle negative sizes ;-)
    sign = size_bytes / math.abs(size_bytes)
    size_bytes = math.abs(size_bytes)

    # Find out 1024-base power (bytes size unit)
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)

    # Handle cases where size_bytes is bigger than 1024 x maximal size unit
    if i < 0 or i >= len(size_units):
        scale = math.pow(1024, i - len(size_units) + 1)
        i = len(size_units) - 1
        p = math.pow(1024, len(size_units) - 1)
    else:
        scale = 1

    value, unit = (scale * sign * float(size_bytes) / p, size_units[i])
    return f'{value:.2f}{unit}' if format_to_str else (value, unit)


def is_roughtly_constant(values: Sequence[Number], threshold: float = 0.01) -> bool:
    return max(values) - min(values) < threshold * sum(map(math.abs, values)) / float(len(values))


def yolo(self: DataCatalog, *search_terms):
    """you only load once, catalog loading helper"""
    return SimpleNamespace(**{
        d: self.load(d)
        for d in self.query(*search_terms)
    })


# Mokey patch catalog yolo loading :-) (code from https://waylonwalker.com/notes/kedro/)
DataCatalog.yolo = yolo
DataCatalog.yolo.__doc__ = "You Only Load Once. (Queries and loads from given search terms)"


def get_by_identifier(identifier: str):
    regex = r'[\w\.]*\w'
    if re.fullmatch(regex, identifier):
        *module_str, name = identifier.split('.')
        module_str = '.'.join(module_str)
        if module_str:
            module = importlib.import_module(module_str)
            return module.__getattribute__(name)
        elif name in globals():
            return globals()[name]
        elif name in locals():
            return locals()[name]
        raise RuntimeError(f'Error: can\'t find ``{identifier}`` identifier (you may have to specify its module)')
    else:
        raise ValueError(f'Error: bad identifier given in `deepcv.utils.get_by_identifier` function (identifier="{identifier}" must match "{regex}" regex)')


def get_str_repr(fn_or_type: Union[Type, Callable], src_file: Optional[str] = None):
    src_file_without_suffix = '.'.join(Path(src_file).parents + [Path(src_file).stem, ]) + '.' if src_file else ''
    signature = inspect.signature(fn_or_type) if isinstance(fn_or_type, Callable) else ''
    return f'`{src_file_without_suffix}{fn_or_type.__name__}{signature}`'


def source_dir(source_file: str = __file__) -> Path:
    return Path(os.path.dirname(os.path.realpath(source_file)))


def try_import(module, log_warn: bool = True, catch_excepts: bool = True) -> Optional[types.ModuleType]:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        msg = f'Couldn\'t import "{module}" module, exception raised: {e}'
        if log_warn:
            logging.warning(f'Warning: {msg}')
        if catch_excepts:
            return None
        raise ImportError(f'Error: {msg}') from e


def import_pickle() -> types.ModuleType:
    """ Returns cPickle module if available, returns imported pickle module otherwise """
    pickle = try_import('cPickle')
    if not pickle:
        pickle = importlib.import_module('pickle')
    return pickle


def import_and_reload(module_name: str, path: Union[str, Path] = Path('.')) -> types.ModuleType:
    """ Import and reload Python source file. Usefull in cases where you need to import your own Python modules from a Jupyter notebook.
    Appends given `path` to Python path, imports and reload given Python module with `importlib` in order to take into account any code modifications.
    NOTE: `import_and_reload` is only intented to be used in ipython or jupyter notebooks, not in production code.
    Returns imported Python module. """
    if path not in sys.path:
        sys.path.append(path)
    module = importlib.import_module(module_name)
    module = importlib.reload(module)
    return module


def import_third_party(src_path: Union[str, Path], namespace: Optional[str] = None, catch_excepts: bool = False) -> Optional[types.ModuleType]:
    """ Imports Python third party modules with importlib. Use this function if you need to import Python modules which are not installed in Python path and outside of deepcv (e.g. can be used to import any git sub-repository projects in `DeepCV/third_party/` directory or to import `DeepCV/src/tests`, which is outside of `DeepCV/src/deepcv` module).
    NOTE: `import_third_party` function won't append third party module path to the system PATH, thus, `import_third_party` is a proper way to import third party modules from code than `import_and_reload`, as there could be conflicts/unexpected bedhaviors when adding third party module path to system PATH. `import_and_reload` is only intented to be used in ipython or jupyter notebooks, not in production code.
    Args:
        src_path: Python third party module's path
        namespace: Namespace/full-name of third party module to import. If `None`, defaults to `path`'s stem (i.e. final component without its eventual suffix)
        catch_excepts: Boolean indicating whether if exceptions should be catched if importlib can't import Python third party module or not. (defaults to `False`)
    Returns imported third party module, or `None` if `catch_excepts` is `True` and third party module couldn't be imported.
    """
    def _import_third_party():
        if not issubclass(src_path, Path):
            src_path = Path(src_path)
        if namespace is None:
            namespace = src_path.stem
        loader = importlib.machinery.SourceFileLoader(namespace, src_path)
        third_party = loader.load_module(namespace)
        return third_party

    if catch_excepts:
        try:
            return _import_third_party()
        except Exception as e:
            logging.warn(f'Warning: Couldn\'t import third party module from given path: "{src_path}" and namespace/full-name "{namespace}". Exception raised: "{e}".')
            return None
    else:
        return _import_third_party()


import_tests = partial(import_third_party, src_path=source_dir(__file__) / '../tests', namespace='tests')


######### TESTING #########


def test_get_by_identifier():
    fn = get_by_identifier('human_readable_size')
    assert fn.__name__ == 'human_readable_size'
    fn = get_by_identifier('re.match')
    assert fn.__name__ == 'match'
    obj = get_by_identifier('pathlib.Path')
    assert obj.__name__ == 'Path'
    assert obj('./') == Path('./')


def test_import_and_reload():
    pathlib = import_and_reload('pathlib')
    assert pathlib is not None
    pickle = import_pickle()
    assert pickle is not None


def test_source_dir():
    path = source_dir(__file__)
    assert path.exists()
    assert path.is_file()
    assert str(path).endswith('utils.py')


if __name__ == "__main__":
    cli = import_tests().test_module_cli(__file__)
    cli()
