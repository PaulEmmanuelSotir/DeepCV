#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Datasets meta module - datasets.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import threading
import collections
import functools as fn
from pathlib import Path
from typing import Optional, Type, Union, Iterable, Dict, Any

import PIL
import kedro.io
import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets

import deepcv
from ....tests.tests_utils import test_module

__all__ = ['PytorchDatasetWarper', 'ImageDataset']
__author__ = 'Paul-Emmanuel Sotir'

# path.glob("*.[jpg|jpeg]")

TORCHVISION_DATASETS = {v.__class__.__name__: v for n, v in torchvision.datasets.__all__ if issubclass(v, torch.utils.data.Dataset)}


class PytorchDatasetWarper(kedro.io.AbstractDataSet):
    def __init__(self, torch_dataset: Type[torch.utils.data.Dataset], dataset_kwargs: Dict[str, Any]):
        super(PytorchDatasetWarper, self).__init__()
        self.pytorch_dataset = eval(torch_dataset)(**dataset_kwargs)

    def _load(self) -> None: pass
    def _save(self, dataloader: torch.utils.data.DataLoader) -> None: pass

    def _describe(self) -> Dict[str, Any]:
        return {'dataloader': self.dataloader} + self.__dict__


# TODO: apply out_transform to targets
# TODO: refactor ImageIterator and ImageDataset (move some operator overload/methods from iterator to dataset iterable)

class ImageDataset(kedro.io.AbstractVersionedDataSet):
    """ Custom image dataset, for usage outside of pytorch dataset tooling or if versionning is needed.
    TODO: kedro.io.CachedDataset ??
    TODO: keep in mind self._glob_function and self._exists_function
    TODO: implement _exists function for versioning
    """

    def __init__(self, dataset_path: Union[str, Path], version: Optional[kedro.io.Version], in_transform=None, out_transform=None, cache: bool = False, shuffle: bool = True, exists_fn: Callable[[str], bool] = None, glob_fn: Callable[[str], List[str]] = None):
        super(VersionnedImageDataset).__init__(self, dataset_path, version, exists_fn, glob_fn)
        # TODO: parse image folder and targets
        # TODO: refactor storage + remove useless TypeVar-s
        raise NotImplementedError

        self.images_refs = {}
        self.loaded_images = {}  # {path: (image, target) for path, target in self.images_refs}
        self.update(dict(*args, **kwargs))
        self.in_transform, self.out_transform = in_transform, out_transform
        self.cache = cache
        self.shuffle = shuffle

    def get_last_save_version(self) -> Optional[str]:
        pass

    def get_last_load_version(self) -> Optional[str]:
        pass

    def _load(self) -> ImageDataset:
        load_path = self._get_load_path()
        raise NotImplementedError

    def _save(self, data: Any) -> None:
        """Saves image data to the specified filepath"""
        save_path = self._get_save_path()
        raise NotImplementedError

    def _describe(self) -> str:
        """Returns a dict that describes the attributes of the dataset"""
        return """ TODO """

    @property
    def shuffle(self) -> bool:
        return self.thread_locals.shuffle if 'shuffle' in self.thread_locals.__dict__ else self._shuffle

    @property.setter
    def set_shuffle(self, shuffle: bool):
        if 'shuffle' in self.thread_locals.__dict__:
            self.thread_locals.shuffle = shuffle
        self._shuffle = shuffle

    @contextmanager
    def threadlocal_shuffle(self, shuffle: bool = True):
        self.thread_locals.shuffle = shuffle
        try:
            yield
        finally:
            if 'shuffle' in self.thread_locals.__dict__:
                del self.thread_locals.shuffle

    def __getitem__(self, key: Union[ImageT, slice]) -> Tuple[LoadedImageT, MetadataT]:
        if issubclass(key, slice):
            return {self._image_transform(img): self.out_transform(target) for img, target in self.store[key]}
        return self._image_transform(key), self.out_transform(self.store[key])

    def __setitem__(self, key: Union[ImageT, slice], target: Union[MetadataT, Iterable[MetadataT]] = None):
        if issubclass(key, ImageT):
            assert issubclass(target, MetadataT), 'Error: cannot assign multiple targets to a single image.'
        elif issubclass(key, slice):
            assert issubclass(target, Iterable[MetadataT]), 'Error: Cannot assign a single target to an image slice'
        self.store[key] = target

    def __delitem__(self, key: ImageT):
        del self.store[key]

    def __iter__(self) -> Iterable[ImageT, MetadataT]:
        it = iter(self.store)  # TODO: map/warp _image_transform to returned iterator or refactor ImageDataset to be an iterator and

        def _warp_tranform_on_iter(iterator_next_fn):
            def _warper(*args, **kwargs):
                x, y = iterator_next_fn()
                return self._image_transform(x), self.out_transform(y)
            return _warper
        it.__next__ = _warp_tranform_on_iter(it.__next__)

        return it

    def __len__(self) -> int:
        return len(self.store)

    def __instancecheck__(self, instance):
        return type(instance) is ImageDataset or type(instance) is dict

    def __subclasscheck__(self, subclass):
        return super(ImageDataset).__subclasscheck__(subclass) or issubclass(subclass, dict)

    def __enter__(self):
        self.load_all()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_all()

    def popitem(self):
        """ Remove and return an item pair from ImageDataset. Raise KeyError is ImageDataset instance is empty. """
        img, target = self.store.popitem()
        return self._image_transform(img), target

    def load_all(self):
        self.store.values = map(load_image, self.store.keys)

    def release_all(self):
        self.store.values = map(release_image, self.store.keys)

    def _image_transform(self, image: ImageT) -> LoadedImageT:
        image = load_image(image)
        if self.in_transform:
            return self.in_transform(image)
        return image

    @staticmethod
    def load_image(image: ImageT, output_type: Type[LoadedImageT] = PIL.Image) -> LoadedImageT:
        if isinstance(image, output_type):
            return image
        else:
            raise NotImplementedError  # TODO: implement

    @staticmethod
    def release_image(image: ImageT, output_type: Type[RefImageT] = Path) -> RefImageT:
        if isinstance(image, output_type):
            return image
        elif isinstance(image, RefImageT):
            # Convert image reference/path to specified output image reference type
            return output_type(image)
        else:
            assert isinstance(image, LoadedImageT), f'Error occured when releasing image from ImageDataset, unrecognized image type: "{type(image)}" (image value: "{image}")'
            # Release image data
            raise NotImplementedError  # TODO: implement

    @staticmethod
    def fromkeys(cls, images_iterable: Iterable[ImageT], default_target: MetadataT = None) -> ImageDataset:
        """ Creates a new ImageDataset instance from given keys mapped to given target (None by default). """
        self = cls()
        for key in images_iterable:
            self.store[key] = default_target
        return self


if __name__ == '__main__':
    test_module(__file__)
