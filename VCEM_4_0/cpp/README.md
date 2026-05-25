# VCEM_4_0 C++ backend

C++/Eigen reimplementation of the VCEM_4_0 solvers, exposed to Python via
pybind11. The notebooks import `bem_cpp` (and future `propagation_cpp`,
`solver_cpp`) instead of the pure-Python `fracture_utils.Ubem.bem_solver`
when the C++ path is opted into.

## Layout

```
cpp/
  CMakeLists.txt         build definition (MSVC + Eigen + pybind11)
  include/               public C++ headers (kelvin.h, bem_solver.h, ...)
  src/                   C++ implementation
  bindings/              pybind11 glue (bem_cpp_module.cpp)
  python/bem_cpp/        Python-side wrapper package; the built .pyd is
                         installed here so `import bem_cpp` just works
  build/                 CMake build directory (gitignored)
```

## Prerequisites

- Visual Studio 2022 Community (MSVC 19.44+) — installed at
  `C:\Program Files\Microsoft Visual Studio\2022\Community\`
- CMake 3.18+ (`C:\Program Files\CMake\bin\cmake.exe`)
- Conda env `vcem_4_0` (Python 3.10.19, numpy, scipy, matplotlib, pybind11)
- Eigen 3.4.0 vendored at `D:\GitHub\Libraries\eigen-3.4.0` (sibling of repo root)

## Build

```powershell
cd D:\GitHub\Fracture\VCEM_4_0\cpp\build
& "C:\Program Files\CMake\bin\cmake.exe" `
    -G "Visual Studio 17 2022" -A x64 `
    -DPython_EXECUTABLE="C:\Users\Owner\anaconda3\envs\vcem_4_0\python.exe" ..
& "C:\Program Files\CMake\bin\cmake.exe" --build . --config Release --target bem_cpp
& "C:\Program Files\CMake\bin\cmake.exe" --install . --config Release
```

The install step copies `bem_cpp.cp310-win_amd64.pyd` into
`python/bem_cpp/`, so notebooks can:

```python
import sys
sys.path.insert(0, r"D:\GitHub\Fracture\VCEM_4_0\cpp\python")
import bem_cpp
```

## Smoke test

```powershell
& "C:\Users\Owner\anaconda3\envs\vcem_4_0\python.exe" -c "
import sys; sys.path.insert(0, r'D:/GitHub/Fracture/VCEM_4_0/cpp/python')
import bem_cpp, numpy as np
assert np.allclose(bem_cpp.add_one(np.arange(4.0)), np.arange(4.0) + 1.0)
print('OK', bem_cpp.__version__)
"
```
