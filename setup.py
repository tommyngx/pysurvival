#! /usr/bin/env python
#
# Updated for Python 3.10 compatibility

import os
import subprocess
from setuptools import setup, Extension, find_packages
import platform

# Version handling
VERSION = '0.2.0'

# Get numpy include directory safely
try:
    import numpy
    numpy_include = numpy.get_include()
except ImportError:
    subprocess.check_call(['pip', 'install', 'numpy'])
    import numpy
    numpy_include = numpy.get_include()

# Package meta-data
NAME = 'pysurvival'
DESCRIPTION = 'Open source package for Survival Analysis modeling'
URL = 'https://www.pysurvival.io'
EMAIL = 'stephane@squareup.com'
AUTHOR = 'steph-likes-git'
LICENSE = "Apache Software License (Apache 2.0)"

# Current Directory
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# Utility functions
def read_long_description():
    """Read long description from file or return short description if file not found"""
    try:
        description_path = os.path.join(CURRENT_DIR, 'LONG_DESCRIPTION.txt')
        with open(description_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return DESCRIPTION  # Fall back to short description

def install_requires():
    with open(os.path.join(CURRENT_DIR, 'requirements.txt'), 'r') as requirements_file:
        requirements = requirements_file.readlines()
    return requirements

def read_version(*file_paths):
    with codecs.open(os.path.join(CURRENT_DIR, *file_paths), 'r') as fp:
        version_file = fp.read()

    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Compiler settings
extra_compile_args = [
    '-std=c++14',
    '-O3',
    '-Wall',
    '-Wno-unused-function',  # Suppress unused function warnings
    '-fPIC',
    '-DPY_LIMITED_API=0x030A0000',
    '-DCYTHON_USE_TYPE_SLOTS=1'
]
if platform.system() == 'Darwin':  # macOS specific flags
    extra_compile_args.extend(['-stdlib=libc++', '-mmacosx-version-min=10.9'])

# Add flags to handle deprecated Unicode APIs
extra_compile_args.extend([
    '-DPy_LIMITED_API=0x030A0000',  # Python 3.10
    '-DCYTHON_UNICODE_WCHAR_T',  # Use modern Unicode APIs
    '-DCYTHON_UNICODE_WIDE',  # Use wide Unicode
])

# Define extensions
ext_modules = [
    Extension(
        name="pysurvival.utils._functions",
        sources=[
            "pysurvival/cpp_extensions/_functions.cpp",
            "pysurvival/cpp_extensions/functions.cpp"
        ],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.utils._metrics",
        sources=[
            "pysurvival/cpp_extensions/_metrics.cpp",
            "pysurvival/cpp_extensions/non_parametric.cpp",
            "pysurvival/cpp_extensions/metrics.cpp",
            "pysurvival/cpp_extensions/functions.cpp"
        ],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.models._survival_forest",
        sources=[
            "pysurvival/cpp_extensions/_survival_forest.cpp",
            "pysurvival/cpp_extensions/survival_forest_data.cpp",
            "pysurvival/cpp_extensions/survival_forest_utility.cpp",
            "pysurvival/cpp_extensions/survival_forest_tree.cpp"
        ],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.models._coxph",
        sources=[
            "pysurvival/cpp_extensions/_coxph.cpp",
            "pysurvival/cpp_extensions/functions.cpp"
        ],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        name="pysurvival.models._svm",
        sources=[
            "pysurvival/cpp_extensions/_svm.cpp"
        ],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
]

# Setup configuration
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=read_long_description(),
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    ext_modules=ext_modules,
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.8.0',
        'pandas>=1.4.0',
        'scikit-learn>=1.0.2',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    license=LICENSE,
    package_data={'': ['*.csv'],},
    extras_require={'tests': ['pytest', 'pytest-pep8',]},
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
)