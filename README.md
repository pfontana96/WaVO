# WaVO

Wave Visual Odometry — Fourier-based visual odometry.

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
pip install -e . -v --config-settings=editable.rebuild=true
```

## Test

```sh
pip install .[test]
pytest
```

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
