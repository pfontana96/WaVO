#pragma once

#include <limits>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>

#include "image/RGBDFrame.hpp"
#include "image/fft.hpp"

namespace wavo {
namespace image {

/// Intermediate registration state kept for offline inspection. Only filled
/// when a register_* method is called with debug=true; since cv::Mat is
/// refcounted this stores references to surfaces already computed, not copies.
struct RegistrationDebugData {
  // Translation stage (the winning branch, for Fourier–Mellin).
  cv::Mat correlation;  ///< fftshifted CV_32F correlation surface.
  Peak peak{};          ///< Its raw peak (height + subpixel shift).
  float overlap = 0.f;  ///< Fraction of jointly valid pixels.
};

struct RegistrationResult {
  cv::Mat affine;  ///< 2x3 CV_32F mapping target -> source pixels.
  float rmse = std::numeric_limits<float>::infinity();

  // Diagnostics.
  float peak = 0.f;            ///< Translation correlation peak height.
  float ncc = 0.f;             ///< Normalized cross-correlation over the overlap.
  float overlap = 0.f;         ///< Fraction of jointly valid pixels.
  int branch = -1;             ///< Fourier–Mellin 180° branch (-1: phase correlation).
  float log_polar_peak = 0.f;  ///< Rotation/scale correlation peak height.

  std::shared_ptr<RegistrationDebugData> debug;  ///< nullptr unless debug was requested.

  cv::Mat inverse_affine() const;
};

class ImageRegistrator {
 public:
  /// Whichever of the two methods scores the lower RMSE.
  static RegistrationResult register_best(const RGBDFrame& source, const RGBDFrame& target,
                                          bool debug = false);

  /// Translation only, from the phase correlation of the windowed DFTs.
  static RegistrationResult register_phase_correlation(const RGBDFrame& source,
                                                       const RGBDFrame& target, bool debug = false);

  /// Rotation + scale from log-polar spectrum correlation, then translation;
  /// both 180°-ambiguous branches are scored and the better one returned.
  static RegistrationResult register_fourier_mellin(const RGBDFrame& source,
                                                    const RGBDFrame& target, bool debug = false);
};

}  // namespace image
}  // namespace wavo
