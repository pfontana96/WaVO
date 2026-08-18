#pragma once

#include <opencv2/core.hpp>

namespace wavo {
namespace image {

/// np.fft.fftshift over the two spatial axes (any channel count, odd sizes ok).
cv::Mat fftshift(const cv::Mat& m);

/// DFT of img * window (CV_32FC2), optionally fftshifted.
cv::Mat compute_fft(const cv::Mat& img, const cv::Mat& window = cv::Mat(), bool shifted = false);

/// Correlation surface (fftshifted, CV_32F) between two CV_32FC2 spectra;
/// its peak is the translation warping src onto target (source is the
/// conjugated operand). With normalize, magnitudes are discarded first
/// (true phase correlation);
/// norm_alpha keeps noise bins from getting the same unit vote as real signal. 0.0 means
/// cross-correlation and 1.0 means pure phase correlation
cv::Mat correlate(const cv::Mat& src_dft, const cv::Mat& target_dft, float norm_alpha = 0.0f);

struct Peak {
  float value;        ///< Correlation height at the integer peak.
  cv::Point2f shift;  ///< (dx, dy) relative to the surface center (w/2, h/2).
};

/// Peak of a correlation surface, with 3-point parabolic subpixel refinement
/// (clamped to ±1 px) unless `subpixel` is false.
Peak extract_peak(const cv::Mat& correlation, bool subpixel = true);

}  // namespace image
}  // namespace wavo
