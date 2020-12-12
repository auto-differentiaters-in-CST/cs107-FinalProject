# -*- coding: utf-8 -*-
"""
    Setup file for autodiffcst.
    Use setup.cfg to configure your project.

    This file was generated with PyScaffold 3.2.3.
    PyScaffold helps you to put up the scaffold of your new Python project.
    Learn more under: https://pyscaffold.org/
"""
import sys

from pkg_resources import VersionConflict, require
import setuptools

try:
    require('setuptools>=38.3')
except VersionConflict:
    print("Error: version of setuptools is too old (<38.3)!")
    sys.exit(1)



if __name__ == "__main__":
# setuptools.setup(use_pyscaffold=True)

# 

    # with open("docs/documentation.md", "r", encoding="utf-8") as fh:
    #     long_description = fh.read()

    with open('requirements.txt') as f:
        required = f.read().splitlines()
    

    setuptools.setup(
        name="autodiffCST", # Replace with your own username
        version="0.0.2",
        author="Xiaohan Yang, Hanwen Zhang, Runting Yang, Max Li",
        author_email="xiaohan_yang@g.harvard.edu, hzhang1@g.harvard.edu, runting_yang@hsph.harvard.edu, manli@fas.harvard.edu",
        description="This package autodiffCST implements automatic differentiation. Users could perform forward mode, and use it for higher order differentiation.",
        # long_description=long_description,
        # long_description_content_type="text/markdown",
        url="https://github.com/auto-differentiaters-in-CST/cs107-FinalProject",
        # packages=setuptools.find_packages(),
        packages=setuptools.find_packages(where="src"),
        package_dir = {'': "src"},
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
        setup_requires=['pytest-runner'],
        install_requires=['numpy','sympy>=1.0'],
        tests_require=['pytest','coverage','pytest-cov'],
        test_suite="tests",
        py_modules = ['AD_vec', 'AD', "admath"]
    
        )