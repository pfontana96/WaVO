#pragma once

#include <cstddef>
#include <vector>

namespace wavo {

/// Periodic 2-D Hann window of shape (rows, cols), row-major.
/// Used to reduce spectral leakage before FFT-based registration.
std::vector<double> hann_window_2d(std::size_t rows, std::size_t cols);

}  // namespace wavo
