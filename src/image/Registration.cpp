#include "image/Registration.hpp"

#include <cmath>

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

struct Spectra {
  cv::Mat dft, square_dft, logpolar_dft;
};

Spectra compute_spectra(const RGBDFrame& frame) {
  Spectra s;
  frame.compute_dfts(s.dft, s.square_dft, s.logpolar_dft);
  return s;
}

RegistrationResult register_phase_correlation_impl(const RGBDFrame& source, const RGBDFrame& target,
                                                   const cv::Mat& source_dft,
                                                   const cv::Mat& target_dft, bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Mat corr = correlate(source_dft, target_dft, 0.0f);
  const Peak peak = extract_peak(corr);

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

RegistrationResult register_fourier_mellin_impl(const RGBDFrame& source, const RGBDFrame& target,
                                                const Spectra& source_spectra,
                                                const Spectra& target_spectra, bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Size sq = target.square_size();
  const cv::Point2f center(sq.width / 2.f, sq.height / 2.f);

  const cv::Mat lp_corr = correlate(source_spectra.logpolar_dft, target_spectra.logpolar_dft, 0.0f);
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

    const cv::Mat corr = correlate(source_rect_dft, target_spectra.square_dft, 0.0f);
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

RegistrationResult ImageRegistrator::register_best(const RGBDFrame& source, const RGBDFrame& target,
                                                   bool debug) {
  const Spectra source_spectra = compute_spectra(source);
  const Spectra target_spectra = compute_spectra(target);
  RegistrationResult pc = register_phase_correlation_impl(source, target, source_spectra.dft,
                                                          target_spectra.dft, debug);
  RegistrationResult fm =
      register_fourier_mellin_impl(source, target, source_spectra, target_spectra, debug);
  return fm.score >= pc.score ? fm : pc;
}

RegistrationResult ImageRegistrator::register_phase_correlation(const RGBDFrame& source,
                                                                const RGBDFrame& target,
                                                                bool debug) {
  return register_phase_correlation_impl(source, target, compute_spectra(source).dft,
                                         compute_spectra(target).dft, debug);
}

RegistrationResult ImageRegistrator::register_fourier_mellin(const RGBDFrame& source,
                                                             const RGBDFrame& target, bool debug) {
  return register_fourier_mellin_impl(source, target, compute_spectra(source),
                                      compute_spectra(target), debug);
}

}  // namespace image
}  // namespace wavo
