#include "image/Registration.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "image/fft.hpp"

namespace wavo {
namespace image {

namespace {

Score score(const cv::Mat& source_gray, const cv::Mat& target_gray, const cv::Mat& affine,
            float min_overlap = 0.25f) {
  cv::Mat warped;
  cv::warpAffine(source_gray, warped, affine, target_gray.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar::all(0));
  const cv::Mat valid = (warped != 0) & (target_gray != 0);

  Score s;
  s.overlap = static_cast<float>(cv::countNonZero(valid)) / static_cast<float>(valid.total());
  if (s.overlap < min_overlap) return s;

  // Masked stats: rmse² = mean(d)² + std(d)²
  cv::Scalar md, sd;
  cv::meanStdDev(warped - target_gray, md, sd, valid);
  s.rmse = static_cast<float>(std::sqrt(md[0] * md[0] + sd[0] * sd[0]));

  return s;
}

RegistrationResult make_result(cv::Mat affine, const Score& s) {
  RegistrationResult r;
  r.affine = std::move(affine);
  r.score = s;
  return r;
}

RegistrationResult register_correlation_impl(RGBDFrame& source, RGBDFrame& target, float norm_alpha,
                                             bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Mat corr = correlate(source.get_or_compute_dft(FrameSpectra::Type::DFT),
                                 target.get_or_compute_dft(FrameSpectra::Type::DFT), norm_alpha);
  const Peak peak = extract_peak(corr, true);

  cv::Mat affine = (cv::Mat_<float>(2, 3) << 1.f, 0.f, peak.shift.x,  //
                    0.f, 1.f, peak.shift.y);
  RegistrationResult r = make_result(affine, score(source.gray(), target.gray(), affine));
  if (debug) {
    r.debug = std::make_shared<RegistrationDebugData>();
    r.debug->correlation = corr;
    r.debug->peak = peak;
  }
  return r;
}

RegistrationResult register_fourier_mellin_impl(RGBDFrame& source, RGBDFrame& target,
                                                float norm_alpha, bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Size sq = target.square_size();
  const cv::Point2f center(sq.width / 2.f, sq.height / 2.f);

  const cv::Mat lp_corr =
      correlate(source.get_or_compute_dft(FrameSpectra::Type::LOGPOLAR_DFT),
                target.get_or_compute_dft(FrameSpectra::Type::LOGPOLAR_DFT), norm_alpha);
  const Peak lp = extract_peak(lp_corr);

  // Map log-polar pixel shifts back to physical rotation and scale.
  const double angle_step = 180.0 / target.n_theta_rows();
  const double rotation_mod_180 = -lp.shift.y * angle_step;
  const double scale = std::exp(-lp.shift.x * (std::log(target.max_log_polar_radius()) / sq.width));

  auto dbg = debug ? std::make_shared<RegistrationDebugData>() : nullptr;

  RegistrationResult best;
  for (int branch = 0; branch < 2; ++branch) {
    const cv::Mat rot_mat =
        cv::getRotationMatrix2D(center, rotation_mod_180 + 180.0 * branch, scale);
    cv::Mat rectified;
    cv::warpAffine(source.square_gray_zero_mean(), rectified, rot_mat, sq, cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar::all(0));
    const cv::Mat source_rect_dft = compute_fft(rectified, source.square_window());

    const cv::Mat corr = correlate(
        source_rect_dft, target.get_or_compute_dft(FrameSpectra::Type::SQUARE_DFT), norm_alpha);
    const Peak peak = extract_peak(corr);

    cv::Mat affine;
    rot_mat.convertTo(affine, CV_32F);
    affine.at<float>(0, 2) += peak.shift.x;
    affine.at<float>(1, 2) += peak.shift.y;

    // Undo the square padding: t += A·offset - offset.
    const cv::Point2f off = target.square_pad_offset();
    affine.at<float>(0, 2) +=
        affine.at<float>(0, 0) * off.x + affine.at<float>(0, 1) * off.y - off.x;
    affine.at<float>(1, 2) +=
        affine.at<float>(1, 0) * off.x + affine.at<float>(1, 1) * off.y - off.y;

    const Score s = score(source.gray(), target.gray(), affine);
    if (branch == 0 || s > best.score) {
      best = make_result(affine, s);
      if (dbg) {
        dbg->correlation = corr;
        dbg->peak = peak;
      }
    }
  }
  if (dbg) {
    best.debug = std::move(dbg);
  }
  return best;
}

}  // namespace

cv::Mat RegistrationResult::inverse_affine() const {
  cv::Mat inv;
  cv::invertAffineTransform(affine, inv);
  return inv;
}

ImageRegistrator::ImageRegistrator(const std::string& type, float norm_alpha)
    : type_(type), norm_alpha_(norm_alpha) {
  if (type != "best" && type != "correlation" && type != "fourier_mellin")
    throw std::invalid_argument("ImageRegistrator: unknown type '" + type +
                                "' (expected best, correlation or fourier_mellin)");
}

ImageRegistrator::ImageRegistrator(const Parameters& params) {
  const Parameters cfg = schema().validate(params);
  *this = ImageRegistrator(cfg.get<std::string>("type"), cfg.get<float>("norm_alpha"));
}

RegistrationResult ImageRegistrator::run(RGBDFrame& source, RGBDFrame& target, bool debug) const {
  if (type_ == "correlation") return register_correlation_impl(source, target, norm_alpha_, debug);
  if (type_ == "fourier_mellin")
    return register_fourier_mellin_impl(source, target, norm_alpha_, debug);
  // "best" (the constructor rejects anything else): keep the better score.
  const RegistrationResult pc = register_correlation_impl(source, target, norm_alpha_, debug);
  const RegistrationResult fm = register_fourier_mellin_impl(source, target, norm_alpha_, debug);
  return fm.score >= pc.score ? fm : pc;
}

}  // namespace image
}  // namespace wavo
