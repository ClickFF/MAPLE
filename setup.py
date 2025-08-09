#!/usr/bin/env python
"""The setup script."""

import os
from setuptools import find_packages, setup, Extension
import sysconfig


setup(
    author="Xujian Wang",
    author_email= "Hsuchein0126@outlook.com",
    description="Machine Learning Potential Aided Structure Optimizer",
    name='malepso',
    include_package_data=True,
    version='0.1',
    python_requires='>=3.9',
    url='https://github.com/ClickFF/MaLePSO',
)