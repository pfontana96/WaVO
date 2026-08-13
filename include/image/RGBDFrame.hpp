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

/// An RGB/depth pair, undistorted at construction, with everything a frame
/// needs for Fourier-based registration precomputed: windowed DFT, square
/// zero-padded DFT and log-polar DFT of its spectrum magnitude.
class RGBDFrame {
 public:
  /// `bgr` must be CV_8UC3, `depth` single-channel with the same size.
  /// Stored pixels are rectified, so intrinsics() exposes the same camera
  /// matrix with zeroed distortion — use those for downstream (de)projection.
  RGBDFrame(const cv::Mat& bgr, const cv::Mat& depth, const CameraIntrinsics& intrinsics);

  const CameraIntrinsics& intrinsics() const { return intrinsics_; }
  const cv::Mat& color() const { return color_; }
  const cv::Mat& depth() const { return depth_; }
  const cv::Mat& gray() const { return gray_; }  // CV_32F
  const cv::Mat& gray_zero_mean() const { return gray_zero_mean_; }
  const cv::Mat& dft() const { return dft_; }  // CV_32FC2
  const cv::Mat& shifted_dft() const { return shifted_dft_; }
  const cv::Mat& square_dft() const { return square_dft_; }
  cv::Point2f square_pad_offset() const { return square_pad_offset_; }
  cv::Size square_size() const { return square_gray_zero_mean_.size(); }
  const cv::Mat& log_polar_dft() const { return log_polar_dft_; }
  float max_log_polar_radius() const { return max_log_polar_radius_; }
  int n_theta_rows() const { return logpolar_.rows; }

  /// Color and depth warped by the 2x3 `rot_mat` into a `dsize` canvas, with
  /// all registration data recomputed on the warped pair.
  RGBDFrame affine_transform(const cv::Mat& rot_mat, cv::Size dsize) const;

  /// Windowed DFT of the square zero-mean gray warped by the 2x3 `rot_mat`.
  cv::Mat rectified_square_dft(const cv::Mat& rot_mat, cv::Size dsize) const;

 private:
  void compute_log_polar();

  CameraIntrinsics intrinsics_;
  cv::Mat color_, depth_, gray_, gray_zero_mean_, hann_window_;
  cv::Mat dft_, shifted_dft_;
  cv::Mat square_gray_zero_mean_, square_window_, square_dft_;
  cv::Point2f square_pad_offset_;
  cv::Mat logpolar_, log_polar_dft_;
  float max_log_polar_radius_ = 0.f;

  std::unique_ptr<image::BaseWindow> window_;
};

}  // namespace image
}  // namespace wavo
