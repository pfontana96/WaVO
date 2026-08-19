#include "odom/FMUmeyamaOdometer.hpp"

namespace wavo {
namespace odom {

FMUmeyamaOdometer::FMUmeyamaOdometer(const Parameters& parameters)
    : FMUmeyamaOdometer(
          image::ImageRegistrator(schema().validate(parameters).scoped("registration"))) {}

cv::Mat FMUmeyamaOdometer::estimate(image::RGBDFrame& prev, image::RGBDFrame& curr, double dt) {
  image::RegistrationResult registration = registrator_.run(prev, curr, debug());
  cv::Mat transform = pointcloud::estimate_pose(prev, curr, registration);
  return transform;
}

}  // namespace odom
}  // namespace wavo
