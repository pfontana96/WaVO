from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from wavo.fft import phase_correlation, extract_peak
from wavo.image import RGBDImage


@dataclass
class _ImageRegistrationResult:

    affine: np.ndarray
    rmse: float
    debug: dict | None

    @staticmethod
    def compute_rmse(
        source: np.ndarray, target: np.ndarray, affine: np.ndarray
    ) -> float:
        assert source.shape == target.shape

        h, w = source.shape

        warped_target = cv2.warpAffine(
            target,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        diff = warped_target.astype(np.float32) - source.astype(np.float32)

        valid = (warped_target != 0) & (source != 0)

        return np.sqrt(np.mean(diff[valid] ** 2))

    def inverse_affine(self) -> np.ndarray:
        return cv2.invertAffineTransform(self.affine)


class ImageRegistrator:

    @staticmethod
    def register_best(source: RGBDImage, target: RGBDImage) -> _ImageRegistrationResult:

        phase_correlation_result = ImageRegistrator.register_phase_correlation(
            source, target
        )
        fourier_mellin_result = ImageRegistrator.register_fourier_mellin(source, target)

        if fourier_mellin_result.rmse < phase_correlation_result.rmse:
            return fourier_mellin_result

        return phase_correlation_result

    @staticmethod
    def register_phase_correlation(source: RGBDImage, target: RGBDImage):

        assert source.frame.shape == target.frame.shape

        corr = phase_correlation(source.frame.dft, target.frame.dft, normalize=True)
        peak, shift = extract_peak(corr, subpixel=True)

        affine = np.zeros((2, 3), dtype=np.float32)
        affine[0, 0] = 1
        affine[1, 1] = 1
        affine[:, 2] = shift

        rmse = _ImageRegistrationResult.compute_rmse(
            source.frame.gray, target.frame.gray, affine
        )

        return _ImageRegistrationResult(affine=affine, rmse=rmse, debug={"peak": peak})

    @staticmethod
    def register_fourier_mellin(source: RGBDImage, target: RGBDImage):

        assert source.frame.shape == target.frame.shape

        h, w = source.frame.square_shape
        center = (w / 2, h / 2)

        self_log_polar_dft, max_radius = source.frame.get_log_polar()
        other_log_polar_dft, _ = target.frame.get_log_polar()

        log_polar_correlation = phase_correlation(
            self_log_polar_dft, other_log_polar_dft, normalize=True
        )
        log_polar_peak, (delta_rho_polar, delta_theta_polar) = extract_peak(
            log_polar_correlation, subpixel=True
        )

        # Map the log-polar pixel shifts back to physical rotation and scale
        # Angle step size per pixel along the vertical axis
        angle_step = 360.0 / h

        d_rho_true, d_theta_true = -delta_rho_polar, -delta_theta_polar
        rotation_angle = d_theta_true * angle_step
        scale_factor = np.exp(d_rho_true * (np.log(max_radius) / w))

        rot_mat = cv2.getRotationMatrix2D(center, rotation_angle, scale_factor)
        target_rect_dft = target.frame.rectified_square_dft(rot_mat, (w, h))

        correlation = phase_correlation(source.frame.square_dft, target_rect_dft)
        peak, shift = extract_peak(correlation, subpixel=True)

        affine_padded = rot_mat.astype(np.float32).copy()
        affine_padded[0, 2] += shift[0]
        affine_padded[1, 2] += shift[1]

        affine = affine_padded.copy()
        affine[:, 2] = (
            affine[:, 2]
            + affine[:2, :2] @ source.frame.square_pad_offset
            - source.frame.square_pad_offset
        )

        rmse = _ImageRegistrationResult.compute_rmse(
            source.frame.gray, target.frame.gray, affine
        )

        return _ImageRegistrationResult(
            affine=affine,
            rmse=rmse,
            debug={"peak": peak, "log_polar_peak": log_polar_peak},
        )
