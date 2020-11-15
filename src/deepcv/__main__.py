#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Entry point when invoked with `python -m <deepcv.run.ProjectContext.package_name>` """  # pragma: no cover

__author__ = 'Paul-Emmanuel Sotir'

def main(): # pragma: no cover
    import sys
    from deepcv.run import run_package, ProjectContext

    if sys.argv[0].endswith('__main__.py'):
        sys.argv[0] = f'python -m {ProjectContext.package_name}'
    
    # Entry point for running a Kedro project throught package's `__main__.py` (e.g. `python -m <package_name>`). Project is either packaged with `kedro package` or is a pip-installed project.
    # and runned by `python -m <package_name>.run` cmd.
    run_package()

if __name__ == '__main__':  # pragma: no cover
    main()
        