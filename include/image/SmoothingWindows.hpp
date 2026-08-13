#pragma once

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace wavo {
namespace image {

// TODO: Add caching to profiles

/// Interface for the smoothing (apodization) windows applied before an FFT.
///
/// Concrete windows only supply their symmetric 1-D profile; the base class
/// assembles both 2-D variants from it:
///  - rectangular(): separable outer-product window (what
///    cv::createHanningWindow produces for the Hann profile).
///  - radial(): the profile tiled over rows, tapering columns only — meant
///    for log-polar images where columns are rho and rows theta.
class BaseWindow {
 public:
  virtual ~BaseWindow() = default;

  /// shape.height x shape.width CV_32F outer product profile(h)ᵀ · profile(w).
  virtual cv::Mat rectangular(cv::Size shape) const {
    return cv::Mat(profile(shape.height).t() * profile(shape.width));
  }

  /// shape.height x shape.width CV_32F tiling of profile(w): tapers columns,
  /// leaves rows untouched.
  virtual cv::Mat radial(cv::Size shape) const {
    return cv::repeat(profile(shape.width), shape.height, 1);
  }

  /// rectangular() weighted by a registration prior: the window is warped by
  /// `prior` (2x3 affine mapping the other frame's pixels into this one) and
  /// multiplied with the unwarped window, so signal expected to be invisible
  /// in the other frame is hidden from the correlation. The product keeps the
  /// taper smooth both at the frame border and at the expected-overlap border
  /// (a warped window alone can hit the frame border at full weight). With an
  /// identity prior this degenerates to the squared window, not the window.
  virtual cv::Mat rectangular(cv::Size shape, const cv::Mat& prior) const {
    const cv::Mat own = rectangular(shape);
    cv::Mat warped;
    cv::warpAffine(own, warped, prior, shape, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar::all(0));
    return own.mul(warped);
  }

 protected:
  /// Symmetric 1 x n CV_32F taper with values in [0, 1].
  virtual cv::Mat profile(int n) const = 0;
};

/// np.hanning: 0.5 * (1 - cos(2*pi*i / (n - 1))). rectangular() matches
/// cv::createHanningWindow; radial() matches the log-polar rho taper.
class HannWindow : public BaseWindow {
 protected:
  cv::Mat profile(int n) const override {
    if (n < 2) return cv::Mat::ones(1, n, CV_32F);
    cv::Mat row(1, n, CV_32F);
    for (int i = 0; i < n; ++i)
      row.at<float>(i) = 0.5f * (1.f - std::cos(2.f * static_cast<float>(CV_PI) * i / (n - 1)));
    return row;
  }
};

/// scipy.signal.windows.tukey: flat top with cosine-tapered edges covering an
/// alpha fraction of the span. alpha = 0 degenerates to rectangular (all
/// ones), alpha = 1 to Hann.
class TukeyWindow : public BaseWindow {
 public:
  explicit TukeyWindow(float alpha) : alpha_(std::clamp(alpha, 0.f, 1.f)) {}

 protected:
  cv::Mat profile(int n) const override {
    cv::Mat row = cv::Mat::ones(1, n, CV_32F);
    if (n < 2 || alpha_ == 0.f) return row;
    const float edge = alpha_ * (n - 1) / 2.f;
    for (int i = 0; i <= static_cast<int>(edge); ++i) {
      const float v =
          0.5f * (1.f + std::cos(static_cast<float>(CV_PI) * (static_cast<float>(i) / edge - 1.f)));
      row.at<float>(i) = v;
      row.at<float>(n - 1 - i) = v;
    }
    return row;
  }

 private:
  float alpha_;
};

}  // namespace image
}  // namespace wavo
