#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "wavo/core.hpp"

namespace py = pybind11;

namespace {

py::array_t<double> hann_window_2d(std::size_t rows, std::size_t cols) {
  auto data = wavo::hann_window_2d(rows, cols);
  py::array_t<double> out({rows, cols});
  std::copy(data.begin(), data.end(), out.mutable_data());
  return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "WaVO C++ core — Fourier-based visual odometry";

  m.def("hann_window_2d", &hann_window_2d, py::arg("rows"), py::arg("cols"),
        "Periodic 2-D Hann window of shape (rows, cols).");

#ifdef WAVO_VERSION_INFO
  m.attr("__version__") = WAVO_VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
