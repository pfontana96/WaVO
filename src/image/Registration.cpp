#include "image/Registration.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

#include "image/fft.hpp"

namespace wavo {
namespace image {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

struct Score {
  float ncc = -kInf, rmse = kInf, overlap = 0.f;
};

/// ncc is -inf when the overlap is too small to trust, which stops a
/// degenerate branch from winning by shrinking the valid region.
Score score(const cv::Mat& source_gray, const cv::Mat& target_gray, const cv::Mat& affine,
            float min_overlap = 0.25f) {
  cv::Mat warped;
  cv::warpAffine(target_gray, warped, affine, source_gray.size(), cv::INTER_LINEAR,
                 cv::BORDER_CONSTANT, cv::Scalar::all(0));
  const cv::Mat valid = (warped != 0) & (source_gray != 0);

  Score s;
  s.overlap = static_cast<float>(cv::countNonZero(valid)) / static_cast<float>(valid.total());
  if (s.overlap < min_overlap) return s;

  // Masked stats: rmse² = mean(d)² + std(d)², ncc = cov(a,b) / (std(a)·std(b)).
  cv::Scalar ma, sa, mb, sb, md, sd;
  cv::meanStdDev(source_gray, ma, sa, valid);
  cv::meanStdDev(warped, mb, sb, valid);
  cv::meanStdDev(warped - source_gray, md, sd, valid);
  s.rmse = static_cast<float>(std::sqrt(md[0] * md[0] + sd[0] * sd[0]));
  if (sa[0] < 1e-6 || sb[0] < 1e-6) return s;

  const double mab = cv::mean(source_gray.mul(warped), valid)[0];
  s.ncc = static_cast<float>((mab - ma[0] * mb[0]) / (sa[0] * sb[0]));
  return s;
}

RegistrationResult make_result(cv::Mat affine, const Score& s, float peak) {
  RegistrationResult r;
  r.affine = std::move(affine);
  r.rmse = s.rmse;
  r.ncc = s.ncc;
  r.overlap = s.overlap;
  r.peak = peak;
  return r;
}

}  // namespace

cv::Mat RegistrationResult::inverse_affine() const {
  cv::Mat inv;
  cv::invertAffineTransform(affine, inv);
  return inv;
}

RegistrationResult ImageRegistrator::register_best(const RGBDFrame& source, const RGBDFrame& target,
                                                   bool debug) {
  RegistrationResult pc = register_phase_correlation(source, target, debug);
  RegistrationResult fm = register_fourier_mellin(source, target, debug);
  return fm.rmse < pc.rmse ? fm : pc;
}

RegistrationResult ImageRegistrator::register_phase_correlation(const RGBDFrame& source,
                                                                const RGBDFrame& target,
                                                                bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Mat corr = phase_correlation(source.dft(), target.dft(), false);
  const Peak peak = extract_peak(corr);

  cv::Mat affine = (cv::Mat_<float>(2, 3) << 1.f, 0.f, peak.shift.x,  //
                    0.f, 1.f, peak.shift.y);
  RegistrationResult r =
      make_result(affine, score(source.gray(), target.gray(), affine), peak.value);
  if (debug) {
    r.debug = std::make_shared<RegistrationDebugData>();
    r.debug->correlation = corr;
    r.debug->peak = peak;
  }
  return r;
}

RegistrationResult ImageRegistrator::register_fourier_mellin(const RGBDFrame& source,
                                                             const RGBDFrame& target, bool debug) {
  CV_Assert(source.gray().size() == target.gray().size());

  const cv::Size sq = source.square_size();
  const cv::Point2f center(sq.width / 2.f, sq.height / 2.f);

  const cv::Mat lp_corr = phase_correlation(source.log_polar_dft(), target.log_polar_dft(), false);
  const Peak lp = extract_peak(lp_corr);

  // Map log-polar pixel shifts back to physical rotation and scale.
  const double angle_step = 180.0 / source.n_theta_rows();
  const double rotation_mod_180 = -lp.shift.y * angle_step;
  const double scale = std::exp(-lp.shift.x * (std::log(source.max_log_polar_radius()) / sq.width));

  auto dbg = debug ? std::make_shared<RegistrationDebugData>() : nullptr;

  RegistrationResult best;
  for (int branch = 0; branch < 2; ++branch) {
    const cv::Mat rot_mat =
        cv::getRotationMatrix2D(center, rotation_mod_180 + 180.0 * branch, scale);
    const cv::Mat target_rect_dft = target.rectified_square_dft(rot_mat, sq);

    const cv::Mat corr = phase_correlation(source.square_dft(), target_rect_dft, false);
    const Peak peak = extract_peak(corr);

    cv::Mat affine;
    rot_mat.convertTo(affine, CV_32F);
    affine.at<float>(0, 2) += peak.shift.x;
    affine.at<float>(1, 2) += peak.shift.y;

    // Undo the square padding: t += A·offset - offset.
    const cv::Point2f off = source.square_pad_offset();
    affine.at<float>(0, 2) +=
        affine.at<float>(0, 0) * off.x + affine.at<float>(0, 1) * off.y - off.x;
    affine.at<float>(1, 2) +=
        affine.at<float>(1, 0) * off.x + affine.at<float>(1, 1) * off.y - off.y;

    const Score s = score(source.gray(), target.gray(), affine);
    if (best.branch < 0 || s.ncc > best.ncc) {
      best = make_result(affine, s, peak.value);
      best.branch = branch;
      best.log_polar_peak = lp.value;
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

}  // namespace image
}  // namespace wavo
