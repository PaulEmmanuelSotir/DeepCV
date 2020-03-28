#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Datasets meta module - datasets.py - `DeepCV`__
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import threading
import collections
import numpy as np
from pathlib import Path
from typing import Optionnal, Type, Union, Iterable

import torch
import torch.nn as nn
import PIL
from kedro.io import AbstractVersionedDataSet

from ....tests.tests_utils import test_module

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

# path.glob("*.[jpg|jpeg]")

LoadedImageT = TypeVar('LoadedImageT', Image, np.array, torch.tensor)
RefImageT = TypeVar('RefImageT', os.file, pathlib.Path, os.PathLike, str)
ImageT = TypeVar('ImageT', RefImageT, LoadedImageT)
MetadataT = Optionnal[dict[str, TypeVar('T')]}


# TODO: apply out_transform to targets
# TODO: refactor ImageIterator and ImageDataset (move some operator overload/methods from iterator to dataset iterable)

class ImageIterator(collections.MutableMapping):
    def __init__(self, images: List[Path], in_transform=None, out_transform=None, cache: bool = False, shuffle: bool = True):
        self.store = {p: ... for p in images}
        self.update(dict(*args, **kwargs))
        self.in_transform, self.out_transform = in_transform, out_transform
        self.cache = cache
        self.shuffle = shuffle

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
            return {self._image_transform(img): target for img, target in self.store[key]}
        return self._image_transform(key), self.store[key]

    def __setitem__(self, key: Union[ImageT, slice], target: Union[MetadataT, Iterable[MetadataT]] = None):
        if issubclass(key, ImageT):
            assert issubclass(target, MetadataT), 'Error: cannot assign multiple targets to a single image.''
        elif issubclass(key, slice):
            assert issubclass(target, Iterable[MetadataT]), 'Error: Cannot assign a single target to an image slice'
        self.store[key] = target

    def __delitem__(self, key: ImageT):
        del self.store[key]

    def __iter__(self) -> Iterable[ImageT, MetadataT]:
        return iter(self.store)  # TODO: map _image_transform to returned iterator or refactor ImageDataset to be an iterator and

    def __len__(self) -> int:
        return len(self.store)

    def __instancecheck__(self, instance):
        return type(instance) is ImageDataset or type(instance) is dict

    def __subclasscheck__(self, subclass):
        return super(ImageDataset).__subclasscheck__(subclass)
        or issubclass(subclass, dict)

    def popitem(self):
        """ Remove and return an item pair from ImageDataset. Raise KeyError is ImageDataset instance is empty. """
        img, target = self.store.popitem()
        return self._image_transform(img), target

    def pushitem(self, image: ImageT, target: MetadataT = None):
        raise NotImplementedError  # TODO: implement

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
            assert isinstance(
                image, LoadedImageT), f'Error occured when releasing image from ImageDataset, unrecognized image type: "{type(image)}" (image value: "{image}")'
            # Release image data
            raise NotImplementedError  # TODO: implement

    @staticmethod
    def fromkeys(cls, images_iterable: Iterable[ImageT], default_target: MetadataT = None) -> ImageDataset:
        """ Creates a new ImageDataset instance from given keys mapped to given target (None by default). """
        self = cls()
        for key in keys_iterable:
            self.store[key] = default_target
        return self


# TODO: return ImageDataset.__iter__() in __iter__ function of VersionnedImageDataset?
class ImageDataset(AbstractVersionedDataSet):
    def __init__(self, , dataset_path: Union[str, Path], version):
        super(VersionnedImageDataset).__init__(self, dataset_path, version)

    @override
    def _load(self) -> ImageDataset:
        load_path = self._get_load_path()
        raise NotImplementedError

    @override
    def _save(self) -> None:
        """Saves image data to the specified filepath"""
        save_path = self._get_save_path()
        raise NotImplementedError

    @override
    def _describe() -> str:
        """Returns a dict that describes the attributes of the dataset"""
        return """ TODO """


if __name__ == '__main__':
    test_module(__file__)
