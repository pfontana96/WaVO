from datetime import date
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from wavo.rgbd_datasets import RGBDDatasetLoader

from wavo._core.pointcloud import estimate_pose
from wavo._core.image import CameraIntrinsics

TUM_INTRINSICS = CameraIntrinsics(
    K=np.array(
        [
            [520.9, 0.0, 325.1],
            [0.0, 521.0, 249.7],
            [0.0, 0.0, 1.0],
        ]
    ),
    dist_coeffs=np.array([0.2312, -0.7849, -0.0033, -0.0001, 0.9172]),
)


if __name__ == "__main__":

    root = "data/rgbd_dataset_freiburg1_desk2"
    dataset = RGBDDatasetLoader.load("tum", root, intrinsics=TUM_INTRINSICS)
    print(dataset.summary())

    gt = np.loadtxt(f"{root}/groundtruth.txt")  # stamp tx ty tz qx qy qz qw

    def gt_pose(stamp):
        r = gt[np.argmin(np.abs(gt[:, 0] - stamp))]
        P = np.eye(4)
        P[:3, :3] = Rotation.from_quat(r[4:8]).as_matrix()
        P[:3, 3] = r[1:4]
        return P

    # seed the estimated trajectory at the ground-truth pose of frame 0
    prev = dataset[0]
    pose = gt_pose(prev.stamp)

    est_traj = [pose[:3, 3]]
    gt_traj = [pose[:3, 3]]
    est_poses = [(prev.stamp, pose.copy())]  # (timestamp, 4x4 world pose)
    times = []

    T = np.eye(4, dtype=np.float32)
    for i in range(1, len(dataset)):
        cur = dataset[i]

        start = perf_counter()
        # frames are undistorted at load time -> use the rectified intrinsics
        T = estimate_pose(prev.rgbd_frame, cur.rgbd_frame, T)
        pose = pose @ np.linalg.inv(T)
        end = perf_counter()
        elapsed = end - start

        # X_cur = T_rel @ X_prev, same convention as estimate_pose returns
        T_rel_gt = np.linalg.inv(gt_pose(cur.stamp)) @ gt_pose(prev.stamp)

        ang = lambda M: np.rad2deg(
            np.linalg.norm(Rotation.from_matrix(M[:3, :3]).as_rotvec())
        )
        print(
            f"frame {i}/{len(dataset) - 1} ({elapsed:.4f} s)"
            f"rot  est {ang(T):6.3f}  gt {ang(T_rel_gt):6.3f}   "
            f"trans est {np.linalg.norm(T[:3,3])*1000:7.2f}mm  gt {np.linalg.norm(T_rel_gt[:3,3])*1000:7.2f}mm  "
            f"ratio {np.linalg.norm(T[:3,3])/max(np.linalg.norm(T_rel_gt[:3,3]),1e-9):.3f}"
        )

        est_traj.append(pose[:3, 3].copy())
        gt_traj.append(gt_pose(cur.stamp)[:3, 3])
        est_poses.append((cur.stamp, pose.copy()))
        times.append(elapsed)
        prev = cur

    est_traj, gt_traj = np.array(est_traj), np.array(gt_traj)
    print(
        f"\nATE RMSE: {np.sqrt(np.mean(np.sum((est_traj - gt_traj) ** 2, axis=1))):.4f} m"
    )
    print(
        f"TIME:\n\tmean: {np.mean(times):.4f} s\n\tmedian: {np.median(times):.4f} s\n\tmax: {np.max(times):.4f} s\n\tmin: {np.min(times):.4f} s"
    )

    # write estimated trajectory in TUM format: timestamp tx ty tz qx qy qz qw
    out_path = f"{date.today().isoformat()}.txt"
    with open(out_path, "w") as f:
        f.write("# estimated trajectory\n")
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for stamp, P in est_poses:
            tx, ty, tz = P[:3, 3]
            qx, qy, qz, qw = Rotation.from_matrix(P[:3, :3]).as_quat()
            f.write(
                f"{stamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
    print(f"Wrote estimated trajectory to {out_path}")

    ax = plt.figure().add_subplot(projection="3d")
    ax.plot(*gt_traj.T, label="ground truth")
    ax.plot(*est_traj.T, label="estimated")
    ax.legend()
    plt.show()
