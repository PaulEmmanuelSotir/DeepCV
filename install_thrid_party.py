#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Training loop meta module - training_loop.py - `DeepCV`__  
.. moduleauthor:: Paul-Emmanuel Sotir
"""
import os
import sys
import click
import logging
from pathlib import Path

import kedro_cli
from kedro.cli import main as kedro_main
import deepcv.utils

__all__ = []
__author__ = 'Paul-Emmanuel Sotir'

THIRD_PARTY_DIR = Path(kedro_cli.PROJ_NAME) / 'src' / 'third_party'
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
OPENCV_ARG_HELP = """ Install OpenCV third party github submodule. """
APEX_ARG_HELP = """ Install NVidia Apex third party github submodule. """
DETECTRON2_ARG_HELP = """ Install Detectron2 third party github submodule. """
IMAGENETV2_ARG_HELP = """ Install ImageNetV2 third party github submodule. """
SINGAN_ARG_HELP = """ Install SinGAN third party github submodule. """


def patch_install(install_cmd):
    def _install_with_third_party(*args, **kwargs):
        rtn = install_cmd(*args, **kwargs)
        install_third_party()
        return rtn
    return _install_with_third_party


kedro_cli.install = patch_install(kedro_cli.install)


@kedro_cli.cli.command()
@click.option('--no--opencv', '-o', 'no_opencv', is_flag=True, multiple=False, default=False, help=OPENCV_ARG_HELP)
@click.option('--no--apex', '-a', 'no_apex', is_flag=True, multiple=False, default=False, help=APEX_ARG_HELP)
@click.option('--no--detectron2', '-d', 'no_detectron2', is_flag=True, multiple=False, default=False, help=DETECTRON2_ARG_HELP)
@click.option('--no--imagenetv2', '-i', 'no_imagenetv2', is_flag=True, multiple=False, default=False, help=IMAGENETV2_ARG_HELP)
@click.option('--no--singan', '-s', 'no_singan', is_flag=True, multiple=False, default=False, help=SINGAN_ARG_HELP)
def install_third_party(no_opencv=False, no_apex=False, no_detectron2=False, no_imagenetv2=False, no_singan=False):
    logging.info('Installing third party git submodules...')
    if not no_opencv:
        install_opencv()
    if not no_apex:
        install_apex()
    if not no_detectron2:
        install_detectron2()
    if not no_imagenetv2:
        install_imagenetv2()
    if not no_singan:
        install_singan()


def install_opencv():
    logging.info('Installing OpenCV third party...')
    with deepcv.utils.cd(THIRD_PARTY_DIR):
        raise NotImplementedError


def install_apex():
    logging.info('Installing NVidia Apex third party...')
    with deepcv.utils.cd(THIRD_PARTY_DIR):
        # TODO: run these comands
        r'git clone https://github.com/NVIDIA/apex'
        with deepcv.utils.cd(r'SinGAN'):
            sub = r'pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" .'
            failed = sub.returncode
            if sys.platform.lower().startswith('win') and failed:
                sub = r'pip install - v - -no-cache-dir --global-option="--pyprof" .'
            if failed:
                logging.error('')
            raise NotImplementedError


def install_detectron2():
    logging.info('Installing Detectron2 third party...')
    with deepcv.utils.cd(THIRD_PARTY_DIR):
        raise NotImplementedError


def install_imagenetv2():
    logging.info('Installing ImageNetV2 third party...')
    with deepcv.utils.cd(THIRD_PARTY_DIR):
        raise NotImplementedError


def install_singan():
    logging.info('Installing SinGAN third party...')
    with deepcv.utils.cd(THIRD_PARTY_DIR):
        # TODO: run these comands
        sub = r'git clone https://github.com/tamarott/SinGAN.git'
        with deepcv.utils.cd(r'SinGAN'):
            sub = r'python -m pip install -r requirements.txt'
            raise NotImplementedError


if __name__ == "__main__":
    os.chdir(str(kedro_cli.PROJ_PATH))
    kedro_main()
