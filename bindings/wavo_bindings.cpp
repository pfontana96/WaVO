#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include "image/RGBDFrame.hpp"
#include "image/Registration.hpp"
#include "image/fft.hpp"
#include "pointcloud/utils.hpp"

namespace py = pybind11;

// numpy <-> cv::Mat, zero-copy both ways: loaded Mats wrap the numpy buffer
// (kept alive for the duration of the call); returned arrays share the Mat's
// refcounted data through a capsule. 1-D arrays load as 1 x N single-channel
// Mats, (H, W) as single-channel, (H, W, C) as C-channel.
namespace pybind11 {
namespace detail {

template <>
struct type_caster<cv::Mat> {
  PYBIND11_TYPE_CASTER(cv::Mat, const_name("numpy.ndarray"));

  bool load(handle src, bool) {
    if (src.is_none()) return true;  // None -> empty Mat
    array a = array::ensure(src, array::c_style);
    if (!a || a.ndim() < 1 || a.ndim() > 3) return false;
    const int depth = cv_depth(a.dtype());
    const int ch = a.ndim() == 3 ? static_cast<int>(a.shape(2)) : 1;
    if (depth < 0 || ch > 4) return false;
    const int rows = a.ndim() == 1 ? 1 : static_cast<int>(a.shape(0));
    const int cols = static_cast<int>(a.shape(a.ndim() == 1 ? 0 : 1));
    value = cv::Mat(rows, cols, CV_MAKETYPE(depth, ch), const_cast<void*>(a.data()));
    ref = a;
    return true;
  }

  static handle cast(const cv::Mat& m, return_value_policy, handle) {
    if (m.empty()) return none().release();
    auto* holder = new cv::Mat(m);  // shares (refcounts) the pixel data
    capsule base(holder, [](void* p) { delete static_cast<cv::Mat*>(p); });
    std::vector<ssize_t> shape{m.rows, m.cols};
    std::vector<ssize_t> strides{static_cast<ssize_t>(m.step[0]),
                                 static_cast<ssize_t>(m.elemSize())};
    if (m.channels() > 1) {
      shape.push_back(m.channels());
      strides.push_back(static_cast<ssize_t>(m.elemSize1()));
    }
    return array(np_dtype(m.depth()), shape, strides, m.data, base).release();
  }

 private:
  array ref;

  static int cv_depth(const dtype& dt) {
    const int n = dt.num();
    if (n == dtype::of<uint8_t>().num()) return CV_8U;
    if (n == dtype::of<int8_t>().num()) return CV_8S;
    if (n == dtype::of<uint16_t>().num()) return CV_16U;
    if (n == dtype::of<int16_t>().num()) return CV_16S;
    if (n == dtype::of<int32_t>().num()) return CV_32S;
    if (n == dtype::of<float>().num()) return CV_32F;
    if (n == dtype::of<double>().num()) return CV_64F;
    return -1;
  }

  static dtype np_dtype(int depth) {
    switch (depth) {
      case CV_8U:
        return dtype::of<uint8_t>();
      case CV_8S:
        return dtype::of<int8_t>();
      case CV_16U:
        return dtype::of<uint16_t>();
      case CV_16S:
        return dtype::of<int16_t>();
      case CV_32S:
        return dtype::of<int32_t>();
      case CV_32F:
        return dtype::of<float>();
      case CV_64F:
        return dtype::of<double>();
      default:
        throw std::invalid_argument("unsupported cv::Mat depth");
    }
  }
};

}  // namespace detail
}  // namespace pybind11

namespace {

using wavo::image::CameraIntrinsics;
using wavo::image::RGBDFrame;

CameraIntrinsics make_intrinsics(const cv::Mat& K, const cv::Mat& dist_coeffs,
                                 float no_valid_point) {
  if (K.rows != 3 || K.cols != 3 || K.channels() != 1)
    throw std::invalid_argument("Camera matrix should be (3, 3)");
  CameraIntrinsics c;
  cv::Mat k32, d32;
  K.convertTo(k32, CV_32F);
  c.K = cv::Matx33f(k32.ptr<float>());
  if (!dist_coeffs.empty()) {
    if (dist_coeffs.total() != 5 || dist_coeffs.channels() != 1)
      throw std::invalid_argument("Only plumb bob model supported (5 coefficients)");
    dist_coeffs.convertTo(d32, CV_32F);
    std::memcpy(c.dist_coeffs.val, d32.ptr<float>(), 5 * sizeof(float));
  }
  c.no_valid_point = no_valid_point;
  return c;
}

cv::Size to_size(std::pair<int, int> wh) { return {wh.first, wh.second}; }

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "WaVO C++ core — Fourier-based visual odometry";

  py::module image = m.def_submodule("image", "RGB-D frames for registration");

  py::class_<CameraIntrinsics>(image, "CameraIntrinsics")
      .def(py::init(&make_intrinsics), py::arg("K"), py::arg("dist_coeffs") = cv::Mat(),
           py::arg("no_valid_point") = 0.f,
           "Pinhole intrinsics with plumb-bob distortion (default: none).")
      .def_property_readonly(
          "K", [](const CameraIntrinsics& c) { return py::array_t<float>({3, 3}, c.K.val); })
      .def_property_readonly(
          "dist_coeffs",
          [](const CameraIntrinsics& c) { return py::array_t<float>(5, c.dist_coeffs.val); })
      .def_readonly("no_valid_point", &CameraIntrinsics::no_valid_point);

  py::class_<RGBDFrame>(image, "RGBDFrame")
      .def(py::init<const cv::Mat&, const cv::Mat&, const CameraIntrinsics&>(), py::arg("bgr"),
           py::arg("depth"), py::arg("intrinsics"),
           "Undistorts bgr (uint8 HxWx3) / depth (HxW) on construction and\n"
           "caches the image-domain registration inputs (zero-mean gray,\n"
           "square padding, smoothing windows). No spectra are stored.")
      .def(
          "compute_dfts",
          [](const RGBDFrame& f) {
            cv::Mat dft, square_dft, logpolar_dft;
            f.compute_dfts(dft, square_dft, logpolar_dft);
            return py::make_tuple(dft, square_dft, logpolar_dft);
          },
          "(dft, square_dft, logpolar_dft) of the frame, all (H, W, 2).")
      .def_property_readonly("intrinsics", &RGBDFrame::intrinsics)
      .def_property_readonly("color", &RGBDFrame::color)
      .def_property_readonly("depth", &RGBDFrame::depth)
      .def_property_readonly("gray", &RGBDFrame::gray)
      .def_property_readonly("gray_zero_mean", &RGBDFrame::gray_zero_mean)
      .def_property_readonly("square_gray_zero_mean", &RGBDFrame::square_gray_zero_mean)
      .def_property_readonly("square_window", &RGBDFrame::square_window)
      .def_property_readonly(
          "shape", [](const RGBDFrame& f) { return py::make_tuple(f.gray().rows, f.gray().cols); })
      .def_property_readonly("square_shape",
                             [](const RGBDFrame& f) {
                               return py::make_tuple(f.square_size().height, f.square_size().width);
                             })
      .def_property_readonly("square_pad_offset",
                             [](const RGBDFrame& f) {
                               const float v[2] = {f.square_pad_offset().x,
                                                   f.square_pad_offset().y};
                               return py::array_t<float>(2, v);
                             })
      .def_property_readonly("max_log_polar_radius", &RGBDFrame::max_log_polar_radius)
      .def_property_readonly("n_theta_rows", &RGBDFrame::n_theta_rows)
      .def(
          "affine_transform",
          [](const RGBDFrame& f, const cv::Mat& rot_mat, std::pair<int, int> dsize) {
            return f.affine_transform(rot_mat, to_size(dsize));
          },
          py::arg("rot_mat"), py::arg("dsize"),
          "Warp color+depth by the 2x3 rot_mat into a (width, height) canvas.");

  py::module fft = image.def_submodule("fft", "Spectra and phase correlation");
  fft.def("fftshift", &wavo::image::fftshift, py::arg("m"),
          "np.fft.fftshift over the two spatial axes.");
  fft.def("compute_fft", &wavo::image::compute_fft, py::arg("img"), py::arg("window") = cv::Mat(),
          py::arg("shifted") = false, "DFT of img * window as an (H, W, 2) array.");
  fft.def("phase_correlation", &wavo::image::phase_correlation, py::arg("src_dft"),
          py::arg("target_dft"), py::arg("normalize") = true, py::arg("eps_rel") = 1e-3f,
          "Correlation surface (fftshifted) between two spectra; eps_rel\n"
          "regularizes the whitening against noise-dominated bins.");
  py::class_<wavo::image::Peak>(fft, "Peak", "Correlation peak (height + subpixel shift).")
      .def_readonly("value", &wavo::image::Peak::value)
      .def_property_readonly(
          "shift", [](const wavo::image::Peak& p) { return py::make_tuple(p.shift.x, p.shift.y); },
          "(dx, dy) relative to the surface center.");
  fft.def(
      "extract_peak",
      [](const cv::Mat& correlation, bool subpixel) {
        const wavo::image::Peak p = wavo::image::extract_peak(correlation, subpixel);
        return py::make_tuple(p.value, py::make_tuple(p.shift.x, p.shift.y));
      },
      py::arg("correlation"), py::arg("subpixel") = true,
      "(peak_value, (dx, dy)) with the shift relative to the surface center.");

  py::module reg = image.def_submodule("registration", "Frame-to-frame registration");

  using wavo::image::ImageRegistrator;
  using wavo::image::RegistrationDebugData;
  using wavo::image::RegistrationResult;
  using wavo::image::Score;

  py::class_<RegistrationDebugData, std::shared_ptr<RegistrationDebugData>>(
      reg, "RegistrationDebugData", "Correlation surfaces and raw peaks; see debug=True.")
      .def_readonly("correlation", &RegistrationDebugData::correlation)
      .def_readonly("peak", &RegistrationDebugData::peak);

  py::class_<Score>(reg, "Score", "Registration quality; ordering compares rmse only.")
      .def_readonly("rmse", &Score::rmse)
      .def_readonly("overlap", &Score::overlap)
      .def(py::self < py::self)
      .def(py::self > py::self)
      .def(py::self <= py::self)
      .def(py::self >= py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__repr__", [](const Score& s) {
        return "Score(rmse=" + std::to_string(s.rmse) + ", overlap=" + std::to_string(s.overlap) +
               ")";
      });

  py::class_<RegistrationResult>(reg, "RegistrationResult")
      .def_readonly("affine", &RegistrationResult::affine)
      .def_readonly("score", &RegistrationResult::score)
      .def_readonly("debug", &RegistrationResult::debug)
      .def("inverse_affine", &RegistrationResult::inverse_affine);

  py::class_<ImageRegistrator>(reg, "ImageRegistrator")
      .def_static("register_best", &ImageRegistrator::register_best, py::arg("source"),
                  py::arg("target"), py::arg("debug") = false)
      .def_static("register_phase_correlation", &ImageRegistrator::register_phase_correlation,
                  py::arg("source"), py::arg("target"), py::arg("debug") = false)
      .def_static("register_fourier_mellin", &ImageRegistrator::register_fourier_mellin,
                  py::arg("source"), py::arg("target"), py::arg("debug") = false);

  py::module pointcloud = m.def_submodule("pointcloud", "Point-cloud utilities");

  // forcecast instead of the cv::Mat caster: pixel coords commonly arrive as
  // integer grids (e.g. np.mgrid -> int64), which the Mat caster rejects.
  pointcloud.def(
      "deproject",
      [](py::array_t<float, py::array::c_style | py::array::forcecast> uv_points,
         py::array_t<float, py::array::c_style | py::array::forcecast> z,
         const CameraIntrinsics& intrinsics) {
        if (uv_points.ndim() != 2) throw std::invalid_argument("uv_points must be (N, 2)");
        if (z.ndim() != 1 || z.shape(0) != uv_points.shape(0))
          throw std::invalid_argument("z must be 1-D with one depth per uv point");
        const cv::Mat uv(static_cast<int>(uv_points.shape(0)), static_cast<int>(uv_points.shape(1)),
                         CV_32F, const_cast<float*>(uv_points.data()));
        const cv::Mat zm(static_cast<int>(z.shape(0)), 1, CV_32F, const_cast<float*>(z.data()));
        return wavo::pointcloud::deproject(uv, zm, intrinsics);
      },
      py::arg("uv_points"), py::arg("z"), py::arg("intrinsics"),
      "Back-project (N, 2) pixel coords with N depths into (N, 3)\n"
      "camera-frame points, undistorting through the intrinsics.");

  pointcloud.def(
      "find_dense_correspondences_3d",
      [](const RGBDFrame& source, const RGBDFrame& target, int stride,
         float min_grad) -> py::tuple {
        const wavo::pointcloud::DenseCorrespondences c =
            wavo::pointcloud::find_dense_correspondences_3d(source, target, stride, min_grad);
        // The Mat caster maps empty Mats to None; keep the (0, 3) shape instead.
        if (c.xyz_source.empty()) {
          const std::vector<py::ssize_t> shape{0, 3};
          return py::make_tuple(py::array_t<float>(shape), py::array_t<float>(shape),
                                py::array_t<uint8_t>(shape), py::array_t<uint8_t>(shape));
        }
        return py::make_tuple(c.xyz_source, c.xyz_target, c.bgr_source, c.bgr_target);
      },
      py::arg("source"), py::arg("target"), py::arg("stride") = 6, py::arg("min_grad") = 8.f,
      "(xyz_source, xyz_target, bgr_source, bgr_target) sampled on a\n"
      "stride-spaced grid of textured source pixels after registering the\n"
      "frames, keeping points with valid depth in both.");

  pointcloud.def("estimate_pose", &wavo::pointcloud::estimate_pose, py::arg("source"),
                 py::arg("target"), py::arg("init_guess") = cv::Mat(), py::arg("stride") = 4,
                 "Rigid 4x4 source -> target pose from the dense 3D correspondences\n"
                 "(closed-form point-to-point Umeyama via Open3D).");

#ifdef WAVO_VERSION_INFO
  m.attr("__version__") = WAVO_VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
