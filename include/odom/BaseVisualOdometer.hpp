#pragma once

#include <mutex>
#include <optional>

#include <opencv2/core.hpp>

#include "Parameters.hpp"
#include "image/RGBDFrame.hpp"

namespace wavo {
namespace odom {

/// Relative transform between the last (prev, curr) pair the odometer
/// estimated, stamped with both frame times.
struct StampedTransform {
  cv::Mat transform;                  ///< 4x4 CV_32F; empty until two frames were estimated.
  double t_prev = 0.0, t_curr = 0.0;  ///< Seconds, as passed to add_frame().
};

/// Frame-to-frame odometry base: owns the (prev, curr) frame pair and the
/// timeline; derived classes only implement the relative-motion estimate
/// between the two frames. No pose integration happens here — composing
/// the relative transforms into a trajectory is the caller's job (the SLAM
/// orchestrator, eventually), as is keeping the two methods in lockstep: a
/// pair replaced before get_transform() saw it is never estimated.
///
/// Both public methods are safe to call from different threads. A single
/// mutex serializes everything, estimate() included, so a get_transform()
/// in flight blocks add_frame(); frames cannot be handed out lock-free
/// because RGBDFrame caches its spectra mutably.
class BaseVisualOdometer {
 public:
  virtual ~BaseVisualOdometer() = default;

  BaseVisualOdometer(bool debug) : debug_(debug) {};
  BaseVisualOdometer(const BaseVisualOdometer&) = delete;
  BaseVisualOdometer& operator=(const BaseVisualOdometer&) = delete;

  /// Adopts `frame` (RGBDFrame is move-only) as the current frame.
  /// `timestamp` is the sensor capture time in seconds — pass the source
  /// clock, not the arrival time. Must be strictly increasing across calls;
  /// throws std::invalid_argument otherwise.
  void add_frame(image::RGBDFrame frame, double timestamp);

  /// Relative transform of the newest (prev, curr) pair, computing it first
  /// if it is not estimated yet (lazy, cached — repeated calls without a new
  /// frame return the same result; check the stamps). The transform is
  /// empty until two frames have been added.
  StampedTransform get_transform();

 protected:
  /// Derived classes are constructed from a Parameters slice:
  ///   explicit Derived(const Parameters& params);
  /// validated against their own ParameterSchema (see ImageRegistrator).
  /// The base has no parameters of its own.
  BaseVisualOdometer() = default;

  bool debug() const { return debug_; }

 private:
  /// Relative motion between the two stored frames as a 4x4 CV_32F rigid
  /// transform mapping prev-frame points into curr's camera frame (source ->
  /// target, the pointcloud::estimate_pose convention). `dt` is curr - prev
  /// timestamps, always > 0. Frames are non-const because RGBDFrame caches
  /// spectra on demand. Called with the internal mutex held.
  virtual cv::Mat estimate(image::RGBDFrame& prev, image::RGBDFrame& curr, double dt) = 0;

  std::mutex mutex_;
  std::optional<image::RGBDFrame> prev_, curr_;
  double prev_timestamp_ = 0.0, curr_timestamp_ = 0.0;
  bool pending_ = false;  ///< (prev_, curr_) not estimated yet.
  StampedTransform result_;
  bool debug_ = false;
};

}  // namespace odom
}  // namespace wavo
