#pragma once

#include <opencv2/core.hpp>

#include "image/RGBDFrame.hpp"
#include "image/Registration.hpp"

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

/// Walks a stride-spaced pixel grid of source, mapping pixels into target
/// through `registration.affine`, keeping points whose image gradient exceeds
/// `min_grad`, whose mapped target pixel lands in bounds and whose depth is
/// valid (non-zero) in both frames; the survivors are deprojected through
/// each frame's intrinsics.
DenseCorrespondences find_dense_correspondences_3d(const image::RGBDFrame& source,
                                                   const image::RGBDFrame& target,
                                                   const image::RegistrationResult& registration,
                                                   int stride = 6, float min_grad = 8.f);

/// Rigid source -> target camera-frame pose from the dense 3D correspondences
/// induced by `registration`, solved closed-form by Open3D's point-to-point
/// estimator (Umeyama). Returns the 4x4 CV_32F transform. `init_guess` is a
/// 4x4 pose (empty = identity), forwarded to Open3D; the closed-form solution
/// does not depend on it. Throws when fewer than 3 correspondences survive.
cv::Mat estimate_pose(const image::RGBDFrame& source, const image::RGBDFrame& target,
                      const image::RegistrationResult& registration,
                      const cv::Mat& init_guess = cv::Mat(), int stride = 4);

}  // namespace pointcloud
}  // namespace wavo
