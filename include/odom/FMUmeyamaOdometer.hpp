#pragma once

#include <utility>

#include "Parameters.hpp"
#include "image/Registration.hpp"
#include "odom/BaseVisualOdometer.hpp"
#include "pointcloud/utils.hpp"

namespace wavo {
namespace odom {

class FMUmeyamaOdometer : public BaseVisualOdometer {
 public:
  FMUmeyamaOdometer(image::ImageRegistrator registrator, bool debug = false)
      : BaseVisualOdometer(debug), registrator_(std::move(registrator)) {}

  /// Config contract: the "registration" subspace holds ImageRegistrator's
  /// schema verbatim (see Parameters::scoped()).
  static const ParameterSchema& schema() {
    static const auto s =
        ParameterSchema("fm_umeyama").include("registration", image::ImageRegistrator::schema());
    return s;
  }

  explicit FMUmeyamaOdometer(const Parameters& parameters);

 private:
  cv::Mat estimate(image::RGBDFrame& prev, image::RGBDFrame& curr, double dt) override;

  image::ImageRegistrator registrator_;
};

}  // namespace odom
}  // namespace wavo
