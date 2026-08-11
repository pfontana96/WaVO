#include "pointcloud/utils.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include "image/Registration.hpp"

namespace wavo {
namespace pointcloud {

namespace {

/// Depth image as CV_32F, avoiding the copy when it already is one.
cv::Mat depth_as_float(const cv::Mat& depth) {
  if (depth.depth() == CV_32F) return depth;
  cv::Mat d32;
  depth.convertTo(d32, CV_32F);
  return d32;
}

}  // namespace

cv::Mat deproject(const cv::Mat& uv_points, const cv::Mat& z,
                  const image::CameraIntrinsics& intrinsics) {
  if (uv_points.channels() != 1 || uv_points.cols != 2)
    throw std::invalid_argument("uv_points must be (N, 2)");
  const int n = uv_points.rows;
  if (z.channels() != 1 || static_cast<int>(z.total()) != n)
    throw std::invalid_argument("z must hold one depth per uv point");
  if (n == 0) return cv::Mat(0, 3, CV_32F);

  cv::Mat uv32 = uv_points, z32 = z.reshape(1, n);
  if (uv32.depth() != CV_32F) uv_points.convertTo(uv32, CV_32F);
  if (z32.depth() != CV_32F) z32.convertTo(z32, CV_32F);

  cv::Mat xy;
  cv::undistortPoints(uv32.reshape(2, n), xy, intrinsics.K, intrinsics.dist_coeffs);
  xy = xy.reshape(1, n);

  cv::Mat xyz(n, 3, CV_32F);
  for (int i = 0; i < n; ++i) {
    const float* p = xy.ptr<float>(i);
    const float depth = z32.at<float>(i);
    float* out = xyz.ptr<float>(i);
    out[0] = p[0] * depth;
    out[1] = p[1] * depth;
    out[2] = depth;
  }
  return xyz;
}

DenseCorrespondences find_dense_correspondences_3d(const image::RGBDFrame& source,
                                                   const image::RGBDFrame& target, int stride,
                                                   float min_grad) {
  if (source.gray().size() != target.gray().size())
    throw std::invalid_argument("source and target must have the same shape");
  const int h = source.gray().rows, w = source.gray().cols;

  const image::RegistrationResult registration =
      image::ImageRegistrator::register_best(source, target);
  cv::Mat source_to_target;
  cv::invertAffineTransform(registration.affine, source_to_target);
  const cv::Matx23f A = source_to_target;

  cv::Mat grad;
  if (min_grad > 0) {
    cv::Mat gx, gy;
    cv::Sobel(source.gray(), gx, CV_32F, 1, 0, 3);
    cv::Sobel(source.gray(), gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, grad);
  }

  const cv::Mat depth_source = depth_as_float(source.depth());
  const cv::Mat depth_target = depth_as_float(target.depth());

  std::vector<cv::Point2f> uv_source, uv_target;
  std::vector<float> z_source, z_target;
  std::vector<cv::Vec3b> bgr_source, bgr_target;

  for (int y = 0; y < h; y += stride) {
    for (int x = 0; x < w; x += stride) {
      if (min_grad > 0 && !(grad.at<float>(y, x) > min_grad)) continue;

      const cv::Point2f t(A(0, 0) * x + A(0, 1) * y + A(0, 2), A(1, 0) * x + A(1, 1) * y + A(1, 2));
      if (t.x < 0.f || t.x > w - 1.f || t.y < 0.f || t.y > h - 1.f) continue;

      // std::rint under the default rounding mode is half-to-even (np.rint).
      const int tx = std::clamp(static_cast<int>(std::rint(t.x)), 0, w - 1);
      const int ty = std::clamp(static_cast<int>(std::rint(t.y)), 0, h - 1);

      const float zs = depth_source.at<float>(y, x);
      const float zt = depth_target.at<float>(ty, tx);
      if (zs == 0.f || zt == 0.f) continue;

      uv_source.emplace_back(static_cast<float>(x), static_cast<float>(y));
      uv_target.push_back(t);
      z_source.push_back(zs);
      z_target.push_back(zt);
      bgr_source.push_back(source.color().at<cv::Vec3b>(y, x));
      bgr_target.push_back(target.color().at<cv::Vec3b>(ty, tx));
    }
  }

  const int n = static_cast<int>(uv_source.size());
  DenseCorrespondences out;
  if (n == 0) {
    out.xyz_source = cv::Mat(0, 3, CV_32F);
    out.xyz_target = cv::Mat(0, 3, CV_32F);
    out.bgr_source = cv::Mat(0, 3, CV_8U);
    out.bgr_target = cv::Mat(0, 3, CV_8U);
    return out;
  }
  out.xyz_source =
      deproject(cv::Mat(uv_source).reshape(1, n), cv::Mat(z_source, false), source.intrinsics());
  out.xyz_target =
      deproject(cv::Mat(uv_target).reshape(1, n), cv::Mat(z_target, false), target.intrinsics());
  out.bgr_source = cv::Mat(bgr_source, true).reshape(1, n);
  out.bgr_target = cv::Mat(bgr_target, true).reshape(1, n);
  return out;
}

}  // namespace pointcloud
}  // namespace wavo
