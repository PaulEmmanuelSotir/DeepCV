#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`Tests utils - tests_utils.py -- `DeepCV`__
.. See ``MODULE_DESCRIPTION`` for more details about this module.
"""
import os
import logging
from pathlib import Path
from functools import partial

import click
import pytest
from kedro.cli.utils import KedroCliError, forward_command, CommandCollection


__all__ = ['test_module']
__author__ = 'Paul-Emmanuel Sotir'

######### TESTING UTILS #########


MODULE_DESCRIPTION = """
This module contains pytest-related tools.

Kedro advice: Tests could be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.  

Test can alternatively be put directly in thair respective source scripts (source scripts tests parsed by 'test_all' function using pytest naming convention).  

To run the tests, run ``kedro test`` or directly run ``python ./src/tests/tests_utils.py`` or pytest command. In either cases, you can specify custom pytest cli arguments.
You may also use `` test_module `` function in your covered source scripted like so:  
``` python
from ....tests.tests_utils import test_module

... # Source implementation of your code and their respective tests (following pytest naming convention)

if __name__ == '__main__':
    test_module(__file__) # Allows you to run pytest only on a given source file by calling it: ```
```
.. moduleauthor:: Paul-Emmanuel Sotir
"""


SCRIPT_ARG_HELP = """ Python script to test using pytest """


@click.group(context_settings={'help_option_names': ["-h", "--help"]}, name=__file__)
def tests_cli(*pytest_args, **pytest_kwargs):
    return


@forward_command(tests_cli)
@click.option('--script', '-s', 'script_filename', nargs=1, type=click.Path(readable=True, dir_okay=False, file_okay=True, exists=True), help=SCRIPT_ARG_HELP)
def test(script_filename, args):
    """ Python script test for usage when ``script_filename`` come from click CLI arguments """
    return _pytest_module(script_filename, args)


def test_module(script_filename):
    """ Python script test for usage when ``script_filename`` is known without coming from click CLI arguments """
    return forward_command(tests_cli)(partial(_pytest_module, script_filename=script_filename))


def _pytest_module(script_filename, args=[]):
    # TODO: improve this function
    logging.info(f'Testing Python script "{script_filename}" with pytest...')
    rtn = pytest.main([script_filename, *args])
    if rtn != 0:
        raise KedroCliError(f'Python script pytest returned non-zero exit code: {rtn}')
    logging.info('Testing done.')


@forward_command(tests_cli, forward_help=True)
def test_all(args):
    # TODO: debug/improve this function
    src_dir = Path(os.path.dirname(os.path.realpath(__file__))) / r'..' / r'deepcv'
    for (root, _dirs, files) in os.walk(src_dir):
        logging.info(f'> Testing scripts from {root} sub-module...')
        for script in files:
            if script.lower().endswith(r'.py') and script != '__init__.py':
                modulepath = os.path.join(root, script)
                logging.info(f'> Testing "{modulepath}" script...')

                try:
                    success = test_module(modulepath, args)
                except Exception as e:
                    success = False
                    logging.error(f'  Exception raised during unary testing of "{modulepath}" python script. Catched exception message: "{e}".')
                if success:
                    logging.info(f'  "{modulepath}" script tests passed.')
                else:
                    msg = f'  "{modulepath}" script tests failed!'
                    logging.error(msg)

######### OTHER TESTING LOGICS #########


@pytest.fixture
def project_context():
    """ Pytest fixture definition, Test functions following pytest naming convention should take an argument named after this function to access kedro project context fixture. """
    from src.deepcv.run import ProjectContext
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "DeepCV"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.15.7"


if __name__ == '__main__':
    tests_cli()
    CommandCollection(("Testing commands", [tests_cli]))()
