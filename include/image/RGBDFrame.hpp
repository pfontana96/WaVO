#pragma once

#include <memory>
#include <opencv2/core.hpp>

#include "image/SmoothingWindows.hpp"

namespace wavo {
namespace image {

/// Pinhole intrinsics with plumb-bob distortion (all-zero coeffs = none).
struct CameraIntrinsics {
  cv::Matx33f K;
  cv::Vec<float, 5> dist_coeffs = cv::Vec<float, 5>::all(0.f);
  float no_valid_point = 0.f;
};

/// An RGB/depth pair, undistorted at construction, with every image-domain
/// input to Fourier-based registration cached: zero-mean gray, its square
/// zero-padded copy, the smoothing windows and the log-polar geometry.
/// No spectra are stored — compute_dfts() produces them on demand.
class RGBDFrame {
 public:
  /// `bgr` must be CV_8UC3, `depth` single-channel with the same size.
  /// Stored pixels are rectified, so intrinsics() exposes the same camera
  /// matrix with zeroed distortion — use those for downstream (de)projection.
  RGBDFrame(const cv::Mat& bgr, const cv::Mat& depth, const CameraIntrinsics& intrinsics);

  /// Computes the windowed DFT of the zero-mean gray (`dft`), of its square
  /// zero-padded copy (`square_dft`) and of the log-polar image of the
  /// square spectrum magnitude (`logpolar_dft`), all CV_32FC2.
  void compute_dfts(cv::Mat& dft, cv::Mat& square_dft, cv::Mat& logpolar_dft) const;

  const CameraIntrinsics& intrinsics() const { return intrinsics_; }
  const cv::Mat& color() const { return color_; }
  const cv::Mat& depth() const { return depth_; }
  const cv::Mat& gray() const { return gray_; }  // CV_32F
  const cv::Mat& gray_zero_mean() const { return gray_zero_mean_; }
  const cv::Mat& square_gray_zero_mean() const { return square_gray_zero_mean_; }
  const cv::Mat& square_window() const { return square_window_; }
  cv::Point2f square_pad_offset() const { return square_pad_offset_; }
  cv::Size square_size() const { return square_gray_zero_mean_.size(); }
  float max_log_polar_radius() const { return max_log_polar_radius_; }
  int n_theta_rows() const { return square_gray_zero_mean_.rows; }

  /// Color and depth warped by the 2x3 `rot_mat` into a `dsize` canvas, with
  /// all registration data recomputed on the warped pair.
  RGBDFrame affine_transform(const cv::Mat& rot_mat, cv::Size dsize) const;

 private:
  /// Square padding, windows and log-polar geometry — construction only.
  void prepare_square_geometry();

  CameraIntrinsics intrinsics_;
  cv::Mat color_, depth_, gray_, gray_zero_mean_, rect_window_;
  cv::Mat square_gray_zero_mean_, square_window_, radial_window_;
  cv::Point2f square_pad_offset_;
  float max_log_polar_radius_ = 0.f;

  std::unique_ptr<image::BaseWindow> window_;
};

}  // namespace image
}  // namespace wavo
