#include "wavo/core.hpp"

#include <cmath>

namespace wavo {

namespace {

constexpr double kPi = 3.14159265358979323846;

double hann(std::size_t i, std::size_t n) {
  if (n <= 1) return 1.0;
  return 0.5 * (1.0 - std::cos(2.0 * kPi * static_cast<double>(i) / static_cast<double>(n)));
}

}  // namespace

std::vector<double> hann_window_2d(std::size_t rows, std::size_t cols) {
  std::vector<double> out(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    const double wr = hann(r, rows);
    for (std::size_t c = 0; c < cols; ++c) {
      out[r * cols + c] = wr * hann(c, cols);
    }
  }
  return out;
}

}  // namespace wavo
