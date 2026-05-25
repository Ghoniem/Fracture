"""VCEM_4_0 BEM C++ backend package.

The compiled extension (bem_cpp.*.pyd on Windows, bem_cpp.*.so on Unix) is
installed alongside this __init__.py by the CMake `install` target.  See
VCEM_4_0/cpp/README.md for build instructions.
"""

from .bem_cpp import *  # noqa: F401,F403
from .bem_cpp import __version__  # noqa: F401
