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
    def _score(source_gray, target_gray, affine, min_overlap: float = 0.25):
        """(ncc, rmse, overlap). ncc is -inf when the overlap is too small to trust,
        which stops a degenerate branch from winning by shrinking the valid region."""
        h, w = source_gray.shape
        warped = cv2.warpAffine(
            target_gray,
            affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = np.logical_and(warped != 0, source_gray != 0)
        overlap = float(valid.mean())
        if overlap < min_overlap:
            return -np.inf, np.inf, overlap

        a = source_gray[valid].astype(np.float32)
        b = warped[valid].astype(np.float32)
        rmse = float(np.sqrt(np.mean((b - a) ** 2)))
        sa, sb = a.std(), b.std()
        if sa < 1e-6 or sb < 1e-6:
            return -np.inf, rmse, overlap
        ncc = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        return ncc, rmse, overlap

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

        ncc, rmse, overlap = _ImageRegistrationResult._score(
            source.frame.gray, target.frame.gray, affine
        )

        return _ImageRegistrationResult(
            affine=affine,
            rmse=rmse,
            debug={
                "peak": peak,
                "ncc": ncc,
                "overlap": overlap,
            },
        )

    @staticmethod
    def register_fourier_mellin(source: RGBDImage, target: RGBDImage):

        assert source.frame.shape == target.frame.shape

        h, w = source.frame.square_shape
        center = (w / 2, h / 2)

        self_log_polar_dft, max_radius, n_theta = source.frame.get_log_polar()
        other_log_polar_dft, _, _ = target.frame.get_log_polar()

        log_polar_correlation = phase_correlation(
            self_log_polar_dft, other_log_polar_dft, normalize=True
        )
        log_polar_peak, (delta_rho_polar, delta_theta_polar) = extract_peak(
            log_polar_correlation, subpixel=True
        )

        # Map the log-polar pixel shifts back to physical rotation and scale
        # Angle step size per pixel along the vertical axis
        angle_step = 180.0 / n_theta

        d_rho_true, d_theta_true = -delta_rho_polar, -delta_theta_polar
        rotation_mod_180 = d_theta_true * angle_step
        scale_factor = np.exp(d_rho_true * (np.log(max_radius) / w))

        best = None
        for branch, rotation in enumerate((rotation_mod_180, rotation_mod_180 + 180.0)):
            rot_mat = cv2.getRotationMatrix2D(center, rotation, scale_factor)
            target_rect_dft = target.frame.rectified_square_dft(rot_mat, (w, h))

            correlation = phase_correlation(
                source.frame.square_dft, target_rect_dft, normalize=True
            )
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

            ncc, rmse, overlap = _ImageRegistrationResult._score(
                source.frame.gray, target.frame.gray, affine
            )

            candidate = _ImageRegistrationResult(
                affine=affine,
                rmse=rmse,
                debug={
                    "peak": peak,
                    "log_polar_peak": log_polar_peak,
                    "ncc": ncc,
                    "branch": branch,
                    "overlap": overlap,
                },
            )

            if best is None or ncc > best.debug["ncc"]:
                best = candidate

        return best
