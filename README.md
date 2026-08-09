# WaVO

Wave Visual Odometry — Fourier-based visual odometry.

> [!WARNING]
> **Work in progress.** This is an early-stage research codebase: APIs, module
> layout, and results change frequently and nothing should be considered stable.

A C++ core library exposed to Python via [pybind11](https://github.com/pybind/pybind11),
built with [scikit-build-core](https://github.com/scikit-build/scikit-build-core).

## Layout

```
├── cpp/                  # Pure C++ library (no Python dependency)
│   ├── include/wavo/     # Public headers
│   └── src/              # Implementation
├── bindings/             # pybind11 glue → wavo._core extension module
├── src/wavo/             # Python package (src layout)
├── tests/                # pytest suite
├── CMakeLists.txt
└── pyproject.toml        # Build backend + project metadata
```

## Install

Requires Python ≥ 3.9 and a C++17 compiler. CMake and pybind11 are pulled in
automatically by the build backend.

Native dependencies (currently OpenCV core + imgproc, see
[conanfile.txt](conanfile.txt)) are resolved with [Conan](https://conan.io)
by the [scikit-build-core-conan](https://github.com/wu-vincent/scikit-build-core-conan)
build backend — no manual Conan setup is needed. The first build compiles a
minimal static OpenCV (a few minutes); it is cached in `~/.conan2`, so later
builds are fast.

To consume the C++ library directly (without Python), use the same conanfile:

```sh
conan install . --build=missing -s build_type=Release
cmake --preset conan-release
cmake --build --preset conan-release
```

```sh
pip install .
```

For development (rebuilds the extension automatically on import after C++ changes):

```sh
pip install .[dev]  # puts the build tooling (conan, cmake, ninja…) in the venv
pip install -e . -v --no-build-isolation --config-settings=editable.rebuild=true
pre-commit install
```

(`--no-build-isolation` matters: rebuild-on-import re-runs CMake outside pip,
so the build tooling and the Conan toolchain must persist in your environment.)

Formatting is enforced by pre-commit hooks: `clang-format` (Google-based, see
[.clang-format](.clang-format)) on C/C++ and `black` on Python.

## Test

```sh
pip install .[test]
pytest
```

## Try image registration

[examples/test_registration_interframe.py](examples/test_registration_interframe.py)
registers a random pair of frames from a TUM RGBD sequence (phase correlation and
Fourier–Mellin) and plots the aligned images with their residuals.

Download a [TUM RGBD](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)
sequence (e.g. `freiburg1_desk2`) and extract it under `data/`, then from the
repo root:

```sh
python examples/test_registration_interframe.py
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
