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

```sh
pip install .
```

For development (rebuilds the extension automatically on import after C++ changes):

```sh
pip install -e .[dev] -v --config-settings=editable.rebuild=true
pre-commit install
```

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
