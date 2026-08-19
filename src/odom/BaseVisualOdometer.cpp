#include "odom/BaseVisualOdometer.hpp"

#include <stdexcept>
#include <utility>

namespace wavo {
namespace odom {

void BaseVisualOdometer::add_frame(image::RGBDFrame frame, double timestamp) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (curr_ && timestamp <= curr_timestamp_)
    throw std::invalid_argument("BaseVisualOdometer: timestamps must be strictly increasing");
  prev_ = std::move(curr_);
  prev_timestamp_ = curr_timestamp_;
  curr_.emplace(std::move(frame));
  curr_timestamp_ = timestamp;
  pending_ = prev_.has_value();
}

StampedTransform BaseVisualOdometer::get_transform() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (pending_) {
    result_.transform = estimate(*prev_, *curr_, curr_timestamp_ - prev_timestamp_);
    result_.t_prev = prev_timestamp_;
    result_.t_curr = curr_timestamp_;
    pending_ = false;
  }
  return result_;
}

}  // namespace odom
}  // namespace wavo
