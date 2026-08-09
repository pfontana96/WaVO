import numpy as np
import open3d as o3d
import open3d.core as o3c
import open3d.t.pipelines.registration as treg

from wavo.image import RGBDImage, CameraIntrinsics
from wavo.correspondences import find_dense_correspondences


def estimate_pose(
    source: RGBDImage,
    target: RGBDImage,
    intrinsics: CameraIntrinsics,
    init_guess: np.ndarray = np.eye(4, dtype=np.float32),
):

    xyz_source, xyz_target, _, _ = find_dense_correspondences(
        source, target, intrinsics, stride=4
    )

    pcd_source = o3d.t.geometry.PointCloud()
    pcd_source.point.positions = o3c.Tensor(xyz_source, dtype=o3c.float32)

    pcd_target = o3d.t.geometry.PointCloud()
    pcd_target.point.positions = o3c.Tensor(xyz_target, dtype=o3c.float32)

    correspondences = o3c.Tensor(np.arange(len(xyz_source)))
    estimator = treg.TransformationEstimationPointToPoint()

    transform = estimator.compute_transformation(
        pcd_source, pcd_target, correspondences, o3c.Tensor(init_guess)
    )

    return transform.cpu().numpy()

    # sigma = _pair_sigma(
    #     xyz_source[:,2],
    #     xyz_target[:,2],
    #     intrinsics.K[0,0],
    #     intrinsics.K[1,1]
    # )
    # return ransac_rigid(xyz_source, xyz_target, sigma=sigma, subsample=16000)
