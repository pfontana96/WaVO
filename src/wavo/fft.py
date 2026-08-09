import numpy as np
import cv2


def compute_fft(img: np.ndarray, window: np.ndarray, shifted: bool = False):
    dft = cv2.dft(img * window, flags=cv2.DFT_COMPLEX_OUTPUT)
    if not shifted:
        return dft

    return np.fft.fftshift(dft)


# frequency domain
def phase_correlation(
    src_dft: np.ndarray, target_dft: np.ndarray, normalize: bool = True
):

    cross = cv2.mulSpectrums(src_dft, target_dft, 0, conjB=True)

    if normalize:  # phase correlation: keep phase only
        mag = cv2.magnitude(cross[..., 0], cross[..., 1])
        cross /= mag[..., None] + 1e-12

    corr = cv2.idft(cross, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
    corr = np.fft.fftshift(corr)

    return corr


def extract_peak(correlation: np.ndarray, subpixel: bool = True):

    h, w = correlation.shape
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)

    if not subpixel:
        return correlation[peak_y, peak_x], (
            float(peak_x - w // 2),
            float(peak_y - h // 2),
        )

    def refine(c0, cm, cp):
        den = cm - 2.0 * c0 + cp
        return (
            0.0
            if abs(den) < 1e-12
            else float(np.clip(0.5 * (cm - cp) / den, -1.0, 1.0))
        )

    fx = peak_x + refine(
        correlation[peak_y, peak_x],
        correlation[peak_y, (peak_x - 1) % w],
        correlation[peak_y, (peak_x + 1) % w],
    )

    fy = peak_y + refine(
        correlation[peak_y, peak_x],
        correlation[(peak_y - 1) % h, peak_x],
        correlation[(peak_y + 1) % h, peak_x],
    )

    return correlation[peak_y, peak_x], (fx - w // 2, fy - h // 2)
