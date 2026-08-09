from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c

from wavo.rgbd_datasets import RGBDDatasetLoader, BaseRGBDEntry
from wavo.correspondences import find_dense_correspondences, deproject
from wavo.pose_estimation import estimate_pose
from wavo.image import CameraIntrinsics
from wavo.registration import ImageRegistrator

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


def show_entries(source_entry: BaseRGBDEntry, target_entry: BaseRGBDEntry):

    fig, axs = plt.subplots(2, 2)

    for i, (entry, name) in enumerate(
        zip((source_entry, target_entry), ("source", "target"))
    ):
        axs[i, 0].set_title(f"{name} ({entry.rgb_stamp:.3f} s) frame color image")
        axs[i, 0].imshow(entry.rgbd_image.frame.color)

        axs[i, 1].set_title(f"{name} ({entry.depth_stamp:.3f} s) frame depth image")
        axs[i, 1].imshow(entry.rgbd_image.frame.depth, cmap="viridis")


def show_image_fft(img: np.ndarray, shifted_dft: np.ndarray):

    magnitude = cv2.magnitude(shifted_dft[:, :, 0], shifted_dft[:, :, 1])
    log_magnitude = np.log1p(magnitude)
    phase = cv2.phase(shifted_dft[:, :, 0], shifted_dft[:, :, 1])

    _, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, data, title in zip(
        axes,
        (img, magnitude, log_magnitude, phase),
        ("Image", "Magnitude", "Log magnitude", "Phase"),
    ):
        ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")


def show_registration(imga, imgb, affine, title):

    warped_b = cv2.warpAffine(imgb, affine, (imgb.shape[1], imgb.shape[0]))
    diff = warped_b.astype(np.int16) - imga.astype(np.int16)

    valid = (warped_b != 0) & (imga != 0)
    rmse = np.sqrt(np.mean(diff[valid].astype(np.float64) ** 2))
    ncc = np.corrcoef(imga[valid], warped_b[valid])[0, 1]

    fig, axs = plt.subplots(1, 3, figsize=(6, 5))

    fig.suptitle(f"{title} (residual  RMSE {rmse:.1f}  NCC {ncc:.4f})")

    axs[0].imshow(imga, cmap="gray")
    axs[1].imshow(warped_b, cmap="gray")

    im = axs[2].imshow(np.where(valid, diff, 0), cmap="coolwarm", vmin=-40, vmax=40)

    fig.colorbar(im, ax=axs[2])


if __name__ == "__main__":

    dataset = RGBDDatasetLoader.load(
        "tum", "data/rgbd_dataset_freiburg1_desk2", intrinsics=TUM_INTRINSICS
    )

    # i = 15
    i = np.random.randint(50, 600)

    entry_a = dataset[i]
    entry_b = dataset[i + 3]

    print(f"Stamp difference: {entry_b.stamp - entry_a.stamp}")

    # show_entries(entry_a, entry_b)

    img_a = entry_a.rgbd_image
    img_b = entry_b.rgbd_image

    img_a.frame.plot()
    img_b.frame.plot()

    start = perf_counter()
    phase_corr_result = ImageRegistrator.register_phase_correlation(img_a, img_b)
    end = perf_counter()
    print(f"Phase correlation registration took {end - start:.4f} s")

    show_registration(
        img_a.frame.gray,
        img_b.frame.gray,
        phase_corr_result.affine,
        f"Phase Correlation (peak: {phase_corr_result.debug["peak"]:.4f})",
    )

    start = perf_counter()
    fourier_mellin_result = ImageRegistrator.register_fourier_mellin(img_a, img_b)
    end = perf_counter()
    print(f"Fourier-Mellin registration took {end - start:.4f} s")

    show_registration(
        img_a.frame.gray,
        img_b.frame.gray,
        fourier_mellin_result.affine,
        f"Fourier-Mellin (peak: {fourier_mellin_result.debug["peak"]:.4f})",
    )

    # start = perf_counter()
    # xyz_source, xyz_target, bgr_source, bgr_target = find_dense_correspondences(img_a, img_b, entry_a.intrinsics)
    # end = perf_counter()
    # print(f"Correspondence matching took {end - start:.4f} s")

    # start = perf_counter()
    # T = estimate_pose(img_a, img_b, TUM_INTRINSICS)
    # end = perf_counter()
    # print(f"Pose estimation matching took {end - start:.4f}")

    plt.tight_layout()
    plt.show()

    # Visualize xyz_source, xyz_target
    # pcd_source = o3d.t.geometry.PointCloud()
    # pcd_source.point.positions = o3c.Tensor(xyz_source.reshape(-1, 3))
    # pcd_source.point.colors = o3c.Tensor(bgr_source.reshape(-1, 3))

    # pcd_source_transformed = pcd_source.clone()

    # pcd_source_transformed.transform(T)

    # pcd_target = o3d.t.geometry.PointCloud()
    # pcd_target.point.positions = o3c.Tensor(xyz_target.reshape(-1, 3))
    # pcd_target.point.colors = o3c.Tensor(bgr_target.reshape(-1, 3))

    # o3d.visualization.draw(
    #     [
    #         {"name": "source", "geometry": pcd_source},
    #         {"name": "source_transformed", "geometry": pcd_source_transformed},
    #         {"name": "target", "geometry": pcd_target},
    #     ],
    #     show_ui=True,
    # )
