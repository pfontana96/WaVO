from __future__ import annotations

import cv2
import numpy as np
from pydantic import BaseModel, model_validator, Field, ConfigDict


def deproject(
    uv_points: np.ndarray,
    z: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:

    assert uv_points.ndim == 2
    assert z.ndim == 1 and z.size == uv_points.shape[0]

    z = z.reshape(-1, 1)

    # Ensure correct types for OpenCV
    uv_points = uv_points.astype(
        np.float32
    )  # Created internally, using float32 for performance

    K = intrinsics.K
    if K.dtype != np.float32:
        K = K.astype(np.float32)

    dist_coeffs = intrinsics.dist_coeffs
    if dist_coeffs.dtype != np.float32:
        dist_coeffs = dist_coeffs.astype(np.float32).reshape(-1)

    xy_points = cv2.undistortPoints(uv_points, K, dist_coeffs).reshape(-1, 2)
    xyz = np.hstack((xy_points * z, z))

    return xyz


class CameraIntrinsics(BaseModel):

    K: np.ndarray
    dist_coeffs: np.ndarray = Field(
        ..., default_factory=lambda: np.zeros(5, dtype=np.float32)
    )
    no_valid_point: float | int = 0.0

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate(self) -> CameraIntrinsics:
        if self.K.shape != (3, 3):
            raise ValueError(f"Camera matrix should be (3, 3), got {self.K.shape}")

        if self.dist_coeffs.shape != (5,):
            raise ValueError("Only plumb bob model supported")

        return self


def _radial_hann_window(shape: tuple[int, int]) -> np.ndarray:
    n_theta, n_rho = shape
    return np.tile(np.hanning(n_rho).astype(np.float32), (n_theta, 1))


class _Frame:

    # object that stores everything that a frame needs for its registration
    def __init__(self, bgr: np.ndarray, depth: np.ndarray):

        assert bgr.ndim == 3 and bgr.shape[2] == 3
        assert bgr.shape[:2] == depth.shape

        self._color = np.ascontiguousarray(bgr)
        self._depth = np.ascontiguousarray(depth)

        self._gray = cv2.cvtColor(self._color, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._gray_zero_mean = self._gray - np.median(self._gray)
        self._shape = self._gray.shape

        self._hann_window = cv2.createHanningWindow(self._gray.shape[::-1], cv2.CV_32F)

        # FFTs
        self._dft = cv2.dft(
            self._gray_zero_mean * self._hann_window, flags=cv2.DFT_COMPLEX_OUTPUT
        )
        self._shifted_dft = np.fft.fftshift(self._dft)

        # log-polar
        self._compute_log_polar()

    @property
    def shape(self) -> np.ndarray:
        return self._shape

    @property
    def color(self) -> np.ndarray:
        return self._color

    @property
    def gray(self) -> np.ndarray:
        return self._gray

    @property
    def gray_zero_mean(self) -> np.ndarray:
        return self._gray_zero_mean

    @property
    def depth(self) -> np.ndarray:
        return self._depth

    @property
    def dft(self) -> np.ndarray:
        return self._dft

    @property
    def shifted_dft(self) -> np.ndarray:
        return self._shifted_dft

    @property
    def square_dft(self) -> np.ndarray:
        return self._square_dft

    @property
    def square_pad_offset(self) -> np.ndarray:
        return self._square_pad_offset

    @property
    def square_shape(self) -> np.ndarray:
        return self._square_shape

    @staticmethod
    def _pad_square(
        img: np.ndarray, window: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray]:

        if window is not None:
            assert img.shape == window.shape

        h, w = img.shape

        n = max(h, w)
        n += n % 2

        top, left = (n - h) // 2, (n - w) // 2
        bottom, right = n - h - top, n - w - left  # remainder here -> exactly n x n

        padded = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )

        if window is None:
            return padded, np.array([left, top], dtype=np.float32)  # pad_offset

        padded_window = cv2.copyMakeBorder(
            window, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
        return (padded, padded_window), np.array(
            [left, top], dtype=np.float32
        )  # pad_offset

    def affine_transform(self, rot_mat: np.ndarray, center: tuple[int, int]) -> _Frame:

        warped_bgr = cv2.warpAffine(
            self._color,
            rot_mat,
            center,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        warped_depth = cv2.warpAffine(
            self._depth,
            rot_mat,
            center,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return _Frame(warped_bgr, warped_depth)

    def rectified_square_dft(self, rot_mat, dsize):
        warped = cv2.warpAffine(
            self._square_gray_zero_mean,
            rot_mat,
            dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return cv2.dft(warped * self._square_window, flags=cv2.DFT_COMPLEX_OUTPUT)

    def _compute_log_polar(self):

        (self._square_gray_zero_mean, self._square_window), self._square_pad_offset = (
            self._pad_square(self._gray_zero_mean, self._hann_window)
        )
        self._square_shape = self._square_gray_zero_mean.shape

        self._square_dft = cv2.dft(
            self._square_gray_zero_mean * self._square_window,
            flags=cv2.DFT_COMPLEX_OUTPUT,
        )
        square_shifted_dft = np.fft.fftshift(self._square_dft)

        h, w = self._square_shape

        center = (float(w // 2), float(h // 2))
        self._max_log_polar_radius = 0.7 * min(h, w) / 2.0

        planes = cv2.split(square_shifted_dft)
        mag = cv2.magnitude(planes[0], planes[1])
        mag = np.log1p(mag)

        # # OpenCV flags for semi-log polar mapping
        flags = cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS

        h_full = 2 * h  # oversample theta, then keep half
        logpolar = cv2.warpPolar(
            mag, (w, h_full), center, self._max_log_polar_radius, flags
        )
        logpolar = np.ascontiguousarray(logpolar[: h_full // 2])  # theta in [0, 180)

        self._logpolar = logpolar - np.median(logpolar)
        self._n_theta_rows = self._logpolar.shape[0]

        # logpolar_window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        logpolar_window = _radial_hann_window(self._logpolar.shape)

        self._log_polar_dft = cv2.dft(
            self._logpolar * logpolar_window, flags=cv2.DFT_COMPLEX_OUTPUT
        )

    def get_log_polar(self) -> tuple[np.ndarray, float]:
        return self._log_polar_dft, self._max_log_polar_radius, self._n_theta_rows

    def plot(self):
        import matplotlib.pyplot as plt

        _, axs = plt.subplots(3, 3, figsize=(12, 12))

        re, im = cv2.split(self._shifted_dft)
        log_mag = np.log1p(cv2.magnitude(re, im))
        phase = cv2.phase(re, im)

        axs[0, 0].imshow(cv2.cvtColor(self._color, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("color")

        axs[0, 1].imshow(self._gray, cmap="gray")
        axs[0, 1].set_title("gray")

        axs[0, 2].imshow(self._gray_zero_mean, cmap="gray")
        axs[0, 2].set_title("gray zero-mean")

        axs[1, 0].imshow(self._hann_window, cmap="gray")
        axs[1, 0].set_title("hann window")

        axs[1, 1].imshow(log_mag, cmap="gray")
        axs[1, 1].set_title("shifted dft log magnitude")

        axs[1, 2].imshow(phase, cmap="gray")
        axs[1, 2].set_title("shifted dft phase")

        lp_re, lp_im = cv2.split(np.fft.fftshift(self._log_polar_dft))
        lp_log_mag = np.log1p(cv2.magnitude(lp_re, lp_im))
        lp_phase = phase = cv2.phase(lp_re, lp_im)

        axs[2, 0].imshow(self._logpolar, cmap="gray")
        axs[2, 0].set_title("log-polar")

        axs[2, 1].imshow(lp_log_mag, cmap="gray")
        axs[2, 1].set_title("log-polar dft log magnitude")

        axs[2, 2].imshow(lp_phase, cmap="gray")
        axs[2, 2].set_title("log-polar dft phase")

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])


class RGBDImage:
    """An RGB/depth pair, undistorted at construction with the given intrinsics.

    The stored pixels are rectified, so :attr:`intrinsics` exposes the same
    camera matrix with zeroed distortion coefficients — use those for any
    downstream (de)projection, not the raw ones.
    """

    def __init__(
        self, bgr: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics
    ):
        bgr, depth = self._undistort(bgr, depth, intrinsics)
        self._intrinsics = CameraIntrinsics(
            K=intrinsics.K,
            dist_coeffs=np.zeros(5, dtype=np.float32),
            no_valid_point=intrinsics.no_valid_point,
        )
        self._frame = _Frame(bgr=bgr, depth=depth)

    @staticmethod
    def _undistort(
        bgr: np.ndarray, depth: np.ndarray, intrinsics: CameraIntrinsics
    ) -> tuple[np.ndarray, np.ndarray]:
        if not intrinsics.dist_coeffs.any():
            return bgr, depth

        h, w = depth.shape
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics.K,
            intrinsics.dist_coeffs,
            None,
            intrinsics.K,
            (w, h),
            cv2.CV_32FC1,
        )
        bgr = cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR)
        # nearest for depth: interpolating across depth discontinuities would
        # invent geometry between foreground and background
        depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)
        return bgr, depth

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intrinsics

    @property
    def frame(self) -> _Frame:
        return self._frame
