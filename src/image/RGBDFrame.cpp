#include "image/RGBDFrame.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "image/fft.hpp"

namespace wavo {
namespace image {

namespace {

/// np.median for a continuous CV_32F matrix (averages the two middle values).
float median(const cv::Mat& m) {
  std::vector<float> v(m.begin<float>(), m.end<float>());
  const auto mid = v.begin() + v.size() / 2;
  std::nth_element(v.begin(), mid, v.end());
  if (v.size() % 2 != 0) return *mid;
  return 0.5f * (*mid + *std::max_element(v.begin(), mid));
}

/// Exact np.median for a CV_32F matrix whose values are integers in [0, 255]
/// (gray converted from uint8): a 256-bin histogram beats a partial sort ~30x.
float median_of_u8_values(const cv::Mat& m) {
  int hist[256] = {};
  for (int r = 0; r < m.rows; ++r) {
    const float* p = m.ptr<float>(r);
    for (int c = 0; c < m.cols; ++c) ++hist[static_cast<int>(p[c])];
  }
  // np.median: mean of the sorted values at indices (total-1)/2 and total/2.
  const std::size_t k1 = (m.total() - 1) / 2, k2 = m.total() / 2;
  int v1 = -1;
  std::size_t cum = 0;
  for (int v = 0; v < 256; ++v) {
    cum += hist[v];
    if (v1 < 0 && cum > k1) v1 = v;
    if (cum > k2) return 0.5f * (v1 + v);
  }
  return 0.f;  // unreachable for non-empty input
}

/// log1p of the magnitude of a CV_32FC2 spectrum.
cv::Mat log_magnitude(const cv::Mat& complex_dft) {
  cv::Mat planes[2], mag;
  cv::split(complex_dft, planes);
  cv::magnitude(planes[0], planes[1], mag);
  mag += 1.f;  // keeps the log finite and non-negative
  cv::log(mag, mag);
  return mag;
}

/// Zero-pad `img` centered into an n x n canvas; optionally report the
/// (left, top) pad offset.
cv::Mat pad_square(const cv::Mat& img, int n, cv::Point2f* offset = nullptr) {
  const int top = (n - img.rows) / 2, left = (n - img.cols) / 2;
  cv::Mat out;
  cv::copyMakeBorder(img, out, top, n - img.rows - top, left, n - img.cols - left,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));
  if (offset) *offset = {static_cast<float>(left), static_cast<float>(top)};
  return out;
}

}  // namespace

RGBDFrame::RGBDFrame(const cv::Mat& bgr, const cv::Mat& depth, const CameraIntrinsics& intrinsics)
    : intrinsics_{intrinsics.K, cv::Vec<float, 5>::all(0.f), intrinsics.no_valid_point} {
  CV_Assert(bgr.type() == CV_8UC3 && depth.channels() == 1 && bgr.size() == depth.size());

  // TODO: Formalize this to be parametrizable
  window_ = std::make_unique<image::TukeyWindow>(TukeyWindow(0.75));

  if (cv::norm(intrinsics.dist_coeffs, cv::NORM_L1) > 0.) {
    cv::Mat map_x, map_y;
    cv::initUndistortRectifyMap(intrinsics.K, intrinsics.dist_coeffs, cv::noArray(), intrinsics.K,
                                bgr.size(), CV_32FC1, map_x, map_y);
    cv::remap(bgr, color_, map_x, map_y, cv::INTER_LINEAR);
    // Nearest for depth: interpolating across depth discontinuities would
    // invent geometry between foreground and background.
    cv::remap(depth, depth_, map_x, map_y, cv::INTER_NEAREST);
  } else {
    color_ = bgr.clone();
    depth_ = depth.clone();
  }

  cv::cvtColor(color_, gray_, cv::COLOR_BGR2GRAY);
  gray_.convertTo(gray_, CV_32F);
  gray_zero_mean_ = gray_ - median_of_u8_values(gray_);
  rect_window_ = window_->rectangular(gray_.size());

  prepare_square_geometry();
}

void RGBDFrame::prepare_square_geometry() {
  int n = std::max(gray_.rows, gray_.cols);
  n += n % 2;
  square_gray_zero_mean_ = pad_square(gray_zero_mean_, n, &square_pad_offset_);
  square_window_ = pad_square(rect_window_, n);
  max_log_polar_radius_ = 0.7f * n / 2.f;
  radial_window_ = window_->radial(cv::Size(n, n));
}

void RGBDFrame::compute_dfts(cv::Mat& dft, cv::Mat& square_dft, cv::Mat& logpolar_dft) const {
  dft = compute_fft(gray_zero_mean_, rect_window_);

  square_dft = compute_fft(square_gray_zero_mean_, square_window_);
  const cv::Mat mag = log_magnitude(fftshift(square_dft));

  const int n = square_gray_zero_mean_.rows;
  const cv::Point2f center(static_cast<float>(n / 2), static_cast<float>(n / 2));

  // Semi-log polar mapping: oversample theta 2x, then keep [0, 180).
  const int flags = cv::WARP_POLAR_LOG + cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS;
  cv::Mat logpolar;
  cv::warpPolar(mag, logpolar, cv::Size(n, 2 * n), center, max_log_polar_radius_, flags);
  logpolar = logpolar.rowRange(0, n).clone();
  logpolar -= median(logpolar);

  logpolar_dft = compute_fft(logpolar, radial_window_);
}

RGBDFrame RGBDFrame::affine_transform(const cv::Mat& rot_mat, cv::Size dsize) const {
  cv::Mat warped_bgr, warped_depth;
  cv::warpAffine(color_, warped_bgr, rot_mat, dsize, cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                 cv::Scalar::all(0));
  cv::warpAffine(depth_, warped_depth, rot_mat, dsize, cv::INTER_NEAREST, cv::BORDER_CONSTANT,
                 cv::Scalar::all(0));
  return {warped_bgr, warped_depth, intrinsics_};
}

}  // namespace image
}  // namespace wavo
