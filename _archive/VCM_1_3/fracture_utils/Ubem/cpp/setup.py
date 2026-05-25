from setuptools import setup, Extension
import sys
import os

try:
    import pybind11
    pybind11_include = pybind11.get_include()
except ImportError:
    print("ERROR: pybind11 not found")
    sys.exit(1)

# Get Python library directory
python_lib_dir = os.path.join(sys.prefix, 'libs')

ext_modules = [
    Extension(
        'bem_solver',
        sources=['bem_bindings.cpp', 'bem_solver.cpp'],
        include_dirs=[
            pybind11_include,
            './eigen3',
            '.',
        ],
        library_dirs=[python_lib_dir],  # Add this
        language='c++',
        extra_compile_args=['/std:c++14', '/O2', '/EHsc'] if sys.platform == 'win32' else ['-std=c++14', '-O3'],
    ),
]

setup(
    name='bem_solver',
    version='0.1.0',
    ext_modules=ext_modules,
    zip_safe=False,
)