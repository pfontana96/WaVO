#pragma once

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "Parameters.hpp"
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

constexpr float kInf = std::numeric_limits<float>::infinity();

struct Score {
  float rmse = kInf, overlap = 0.f;

  // Ordering considers rmse only (overlap is ignored for now).
  friend bool operator>(const Score& a, const Score& b) { return a.rmse < b.rmse; }
  friend bool operator<(const Score& a, const Score& b) { return b > a; }
  friend bool operator<=(const Score& a, const Score& b) { return !(a > b); }
  friend bool operator>=(const Score& a, const Score& b) { return !(a < b); }
  friend bool operator==(const Score& a, const Score& b) { return a.rmse == b.rmse; }
  friend bool operator!=(const Score& a, const Score& b) { return !(a == b); }
};

struct RegistrationResult {
  cv::Mat affine;  ///< 2x3 CV_32F mapping source -> target pixels.
  Score score;

  std::shared_ptr<RegistrationDebugData> debug;  ///< nullptr unless debug was requested.

  cv::Mat inverse_affine() const;
};

class ImageRegistrator {
 public:
  /// `type` is "best", "correlation" or "fourier_mellin"; anything else
  /// throws std::invalid_argument.
  explicit ImageRegistrator(const std::string& type, float norm_alpha = 0.f);

  /// Config contract for the Parameters constructor.
  static const ParameterSchema& schema() {
    static const auto s = ParameterSchema("registration")
                              .require<std::string>("type")
                              .optional<float>("norm_alpha", 0.f);
    return s;
  }

  explicit ImageRegistrator(const Parameters& params);

  RegistrationResult run(const RGBDFrame& source, const RGBDFrame& target,
                         bool debug = false) const;

 private:
  std::string type_;
  float norm_alpha_;
};

}  // namespace image
}  // namespace wavo
