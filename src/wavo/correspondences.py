import cv2
import numpy as np

from wavo.image import deproject

# C++
from wavo._core.image.registration import ImageRegistrator
from wavo._core.image import RGBDFrame


def _sample_nearest(img, xy):
    """Nearest-neighbour lookup of img at float pixel coords xy (N,2) as (x,y)."""
    h, w = img.shape[:2]
    xi = np.clip(np.rint(xy[:, 0]), 0, w - 1).astype(np.intp)
    yi = np.clip(np.rint(xy[:, 1]), 0, h - 1).astype(np.intp)

    return img[yi, xi]


def find_dense_correspondences(
    source: RGBDFrame,
    target: RGBDFrame,
    stride: int = 6,
    min_grad: float = 8.0,
):
    assert source.shape == target.shape

    h, w = source.shape

    registration_result = ImageRegistrator.register_best(source, target)
    target_to_source_affine = registration_result.affine
    source_to_target_affine = cv2.invertAffineTransform(target_to_source_affine)

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    uv_source = np.stack([xs.ravel(), ys.ravel()], 1)
    uv_target = (
        uv_source.astype(np.float32) @ source_to_target_affine[:2, :2].T
        + source_to_target_affine[:, 2]
    )

    if min_grad > 0:
        gx = cv2.Sobel(source.gray, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(source.gray, cv2.CV_32F, 0, 1, 3)
        grad = cv2.magnitude(gx, gy)
        textured = _sample_nearest(grad, uv_source) > min_grad
        uv_source, uv_target = uv_source[textured], uv_target[textured]

    ok = (
        (uv_target[:, 0] >= 0)
        & (uv_target[:, 0] <= w - 1)
        & (uv_target[:, 1] >= 0)
        & (uv_target[:, 1] <= h - 1)
    )

    uv_source, uv_target = uv_source[ok], uv_target[ok]

    z_source = source.depth[uv_source[:, 1], uv_source[:, 0]]
    z_target = _sample_nearest(target.depth, uv_target)

    z_valid = np.logical_and(z_source != 0, z_target != 0)

    uv_source = uv_source[z_valid]
    z_source = z_source[z_valid]
    uv_target = uv_target[z_valid]
    z_target = z_target[z_valid]

    xyz_source = deproject(uv_source, z_source, source.intrinsics)
    xyz_target = deproject(uv_target, z_target, target.intrinsics)

    bgr_source = source.color[uv_source[:, 1], uv_source[:, 0]]
    bgr_target = _sample_nearest(target.color, uv_target)

    return xyz_source, xyz_target, bgr_source, bgr_target
