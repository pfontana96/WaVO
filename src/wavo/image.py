import cv2
import numpy as np

from wavo._core.image import CameraIntrinsics


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
