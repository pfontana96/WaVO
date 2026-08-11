#pragma once

#include <opencv2/core.hpp>

#include "image/RGBDFrame.hpp"

namespace wavo {
namespace pointcloud {

/// Back-projects pixel coordinates into camera-frame 3D points.
/// `uv_points` is N x 2 single-channel (pixel coords) and `z` holds the N
/// depths; any numeric depth is accepted and converted to CV_32F. Returns
/// the N x 3 CV_32F matrix of (x, y, z) points.
cv::Mat deproject(const cv::Mat& uv_points, const cv::Mat& z,
                  const image::CameraIntrinsics& intrinsics);

/// Dense 3D point correspondences between two RGB-D frames.
struct DenseCorrespondences {
  cv::Mat xyz_source;  ///< N x 3 CV_32F camera-frame points in source.
  cv::Mat xyz_target;  ///< N x 3 CV_32F camera-frame points in target.
  cv::Mat bgr_source;  ///< N x 3 CV_8U colors at the source pixels.
  cv::Mat bgr_target;  ///< N x 3 CV_8U colors at the target pixels.
};

/// Registers source to target (ImageRegistrator::register_best), then walks a
/// stride-spaced pixel grid of source, keeping points whose image gradient
/// exceeds `min_grad`, whose affine-mapped target pixel lands in bounds and
/// whose depth is valid (non-zero) in both frames; the survivors are
/// deprojected through each frame's intrinsics.
DenseCorrespondences find_dense_correspondences_3d(const image::RGBDFrame& source,
                                                   const image::RGBDFrame& target, int stride = 6,
                                                   float min_grad = 8.f);

}  // namespace pointcloud
}  // namespace wavo
