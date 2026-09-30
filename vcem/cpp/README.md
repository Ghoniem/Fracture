# VCEM C++ backend

C++/Eigen reimplementation of the VCEM hot paths, exposed to Python through the pybind11
module `bem_cpp`:

- the boundary-element solver (`BEMSolver2D`, Kelvin kernels);
- the crack operators (edge-dislocation kernels, `assemble_operator`, the boundary
  operators) and the KKT solve;
- segment-intersection detection for topology updates.

The solvers use it when a run selects `engine="cpp"` (the case workbook's `engine` key).
If the extension cannot be imported, they fall back to the Python reference
implementation.

## Layout

```
cpp/
  CMakeLists.txt         build definition (Eigen + OpenMP + pybind11)
  include/vcem/          public C++ headers (kelvin.h, bem_solver.h, kkt.h, ...)
  src/                   C++ implementation
  bindings/              pybind11 glue (bem_cpp_module.cpp)
  python/bem_cpp/        Python package; the built extension is installed here
  python/*.py            C++-vs-Python agreement tests, benchmarks, diagnostics
  build/                 CMake build directory (gitignored)
```

## Prerequisites

- a C++17 compiler: Visual Studio 2022 (MSVC) on Windows, or GCC/Clang;
- CMake 3.18 or newer;
- Eigen 3.4.0 in `Libraries/eigen-3.4.0/`, a sibling of the repository root
  (`D:\GitHub\Libraries\eigen-3.4.0` for a clone at `D:\GitHub\Fracture`);
- the Python environment that will import the module, with `numpy` and `pybind11`
  installed. The extension is built for one Python version (e.g. `cp314`) and loads only
  in that version.

OpenMP is detected automatically; MSVC's bundled OpenMP 2.0 is enough.

## Build

From `vcem/cpp/`, pointing CMake at the target interpreter. On this workstation that is
the repository's `venv_fracture` (Python 3.14):

```powershell
cd vcem\cpp
..\..\venv_fracture\Scripts\python -m pip install pybind11
& "C:\Program Files\CMake\bin\cmake.exe" -S . -B build -G "Visual Studio 17 2022" -A x64 `
    -DPython_EXECUTABLE="$((Resolve-Path ..\..\venv_fracture\Scripts\python.exe).Path)"
& "C:\Program Files\CMake\bin\cmake.exe" --build build --config Release --target bem_cpp --parallel
& "C:\Program Files\CMake\bin\cmake.exe" --install build --config Release
```

On Linux or macOS, drop `-G`/`-A` and pass `-DPython_EXECUTABLE=$(which python)`.

The install step copies the extension (`bem_cpp.cp314-win_amd64.pyd` here) into
`python/bem_cpp/`. After switching Python version, or after moving the repository,
delete `build/`, because the CMake cache records absolute paths, and remove the old
extension file from `python/bem_cpp/`. Then reconfigure.

## Check the build

```powershell
python -c "import sys; sys.path.insert(0, 'vcem/cpp/python'); import bem_cpp; print(bem_cpp.__version__, 'OpenMP', bem_cpp.openmp_available, bem_cpp.openmp_max_threads(), 'threads')"
```

Then run the C++-vs-Python agreement tests from the repository root. Each prints
`PASSED`:

```powershell
foreach ($t in "kernels","edge_dislocation","assemble_operator","boundary_operators","kkt","solve","intersection","end_to_end_cpp") {
    python "vcem/cpp/python/test_${t}_vs_python.py"
}
```

`bem_cpp.__version__` matches the current VCEM release (`fracture_utils.__version__`).
