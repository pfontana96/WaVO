#include "image/fft.hpp"

#include <algorithm>
#include <cmath>

namespace wavo {
namespace image {

cv::Mat fftshift(const cv::Mat& m) {
  CV_Assert(m.rows > 1 && m.cols > 1);
  const int dy = m.rows / 2, dx = m.cols / 2;
  cv::Mat rows, out;
  cv::vconcat(m.rowRange(m.rows - dy, m.rows), m.rowRange(0, m.rows - dy), rows);
  cv::hconcat(rows.colRange(rows.cols - dx, rows.cols), rows.colRange(0, rows.cols - dx), out);
  return out;
}

cv::Mat compute_fft(const cv::Mat& img, const cv::Mat& window, bool shifted) {
  cv::Mat out;
  cv::dft(window.empty() ? img : img.mul(window), out, cv::DFT_COMPLEX_OUTPUT);
  return shifted ? fftshift(out) : out;
}

cv::Mat phase_correlation(const cv::Mat& src_dft, const cv::Mat& target_dft, bool normalize,
                          float eps_rel) {
  cv::Mat cross;
  cv::mulSpectrums(src_dft, target_dft, cross, 0, /*conjB=*/true);
  if (normalize) {  // keep phase only (regularized: see header on eps_rel)
    cv::Mat planes[2], mag;
    cv::split(cross, planes);
    cv::magnitude(planes[0], planes[1], mag);
    mag += eps_rel * static_cast<float>(cv::mean(mag)[0]) + 1e-12f;
    planes[0] /= mag;
    planes[1] /= mag;
    cv::merge(planes, 2, cross);
  }
  cv::Mat corr;
  cv::idft(cross, corr, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
  return fftshift(corr);
}

Peak extract_peak(const cv::Mat& c, bool subpixel) {
  CV_Assert(c.type() == CV_32F);
  double max_val = 0.;
  cv::Point p;
  cv::minMaxLoc(c, nullptr, &max_val, nullptr, &p);

  const int h = c.rows, w = c.cols;
  float fx = static_cast<float>(p.x), fy = static_cast<float>(p.y);
  if (subpixel) {
    const auto at = [&](int y, int x) {
      return static_cast<double>(c.at<float>((y + h) % h, (x + w) % w));
    };
    const auto refine = [](double c0, double cm, double cp) {
      const double den = cm - 2. * c0 + cp;
      return std::abs(den) < 1e-12 ? 0.f
                                   : static_cast<float>(std::clamp(0.5 * (cm - cp) / den, -1., 1.));
    };
    fx += refine(at(p.y, p.x), at(p.y, p.x - 1), at(p.y, p.x + 1));
    fy += refine(at(p.y, p.x), at(p.y - 1, p.x), at(p.y + 1, p.x));
  }
  return {c.at<float>(p), {fx - w / 2, fy - h / 2}};
}

}  // namespace image
}  // namespace wavo
