from time import perf_counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.core as o3c

from wavo.rgbd_datasets import RGBDDatasetLoader

from wavo._core import Parameters

from wavo._core.image import RGBDFrame, CameraIntrinsics
from wavo._core.image.fft import fftshift
from wavo._core.image.registration import ImageRegistrator, RegistrationResult

from wavo._core.pointcloud import find_dense_correspondences_3d, estimate_pose

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

DEBUG = True


def plot_image_registration(
    source: RGBDFrame,
    target: RGBDFrame,
    result: RegistrationResult,
    title: str,
):

    warped_source = cv2.warpAffine(
        source.color, result.affine, (source.shape[1], source.shape[0])
    )

    nrows = 2 if result.debug is None else 3

    fig, axs = plt.subplots(nrows, 3)
    fig.suptitle(f"{title} (RMSE: {result.score.rmse:.4f})")

    axs[0, 0].set_title("source frame")
    axs[0, 0].imshow(source.color)

    axs[0, 1].set_title("target frame")
    axs[0, 1].imshow(target.color)

    axs[0, 2].set_title("warped source")
    axs[0, 2].imshow(warped_source)

    if result.debug is not None:

        for i, (name, frame) in enumerate(zip(("source", "target"), (source, target))):

            dft, _, _ = frame.compute_dfts()
            re, im = cv2.split(fftshift(dft))
            mag = cv2.magnitude(re, im)
            logmag = cv2.log(mag, mag) + 1.0

            axs[1, i].set_title(f"{name} shifted DFT log-mag")
            axs[1, i].imshow(logmag, cmap="viridis")

        axs[1, 2].set_title("correlation")
        axs[1, 2].imshow(result.debug.correlation, cmap="RdBu_r")

    axs[nrows - 1, 0].set_axis_off()
    axs[nrows - 1, 2].set_axis_off()

    axs[nrows - 1, 1].set_title("target - warped_source difference")
    warped_source_gray = cv2.warpAffine(
        source.gray, result.affine, (source.shape[1], source.shape[0])
    )
    valid = warped_source_gray != 0
    diff = np.zeros(source.gray.shape, dtype=np.float32)
    diff[valid] = target.gray[valid] - warped_source_gray[valid]
    axs[nrows - 1, 1].imshow(diff, cmap="coolwarm")


if __name__ == "__main__":

    parameters_file = Path(__file__).parent / "registration_interframe.yaml"
    params = Parameters.from_yaml(str(parameters_file))

    correlation_registrator = ImageRegistrator(params.scoped("correlation"))
    fourier_mellin_registrator = ImageRegistrator(params.scoped("fourier_mellin"))
    best_registrator = ImageRegistrator(params.scoped("best"))

    dataset = RGBDDatasetLoader.load(
        "tum", "data/rgbd_dataset_freiburg1_desk2", intrinsics=TUM_INTRINSICS
    )

    # i = 2
    i = np.random.randint(50, 600)

    entry_a = dataset[i]
    entry_b = dataset[i + 8]

    frame_a = entry_a.rgbd_frame
    frame_b = entry_b.rgbd_frame

    print(f"Stamp difference: {entry_b.stamp - entry_a.stamp}")

    start = perf_counter()
    correlation_result = correlation_registrator.run(
        entry_a.rgbd_frame, entry_b.rgbd_frame, debug=DEBUG
    )
    end = perf_counter()
    print(f"Cross correlation registration took {end - start:.4f} s")

    plot_image_registration(
        frame_a, frame_b, correlation_result, "Cross-Correlation registration"
    )

    start = perf_counter()
    fourier_mellin_result = fourier_mellin_registrator.run(
        entry_a.rgbd_frame, entry_b.rgbd_frame, debug=DEBUG
    )
    end = perf_counter()
    print(f"Fourier-Mellin registration took {end - start:.4f} s")

    plot_image_registration(
        frame_a, frame_b, fourier_mellin_result, "Fourier-Mellin registration"
    )

    start = perf_counter()
    best_result = best_registrator.run(
        entry_a.rgbd_frame, entry_b.rgbd_frame, debug=DEBUG
    )
    end = perf_counter()
    print(f"Best registration took {end - start:.4f} s")

    plot_image_registration(frame_a, frame_b, best_result, "Best registration")

    start = perf_counter()
    xyz_source, xyz_target, bgr_source, bgr_target = find_dense_correspondences_3d(
        entry_a.rgbd_frame, entry_b.rgbd_frame, best_result
    )
    end = perf_counter()
    print(f"Correspondence matching took {end - start:.4f} s")

    start = perf_counter()
    T = estimate_pose(entry_a.rgbd_frame, entry_b.rgbd_frame, best_result)
    end = perf_counter()
    print(f"Pose estimation matching took {end - start:.4f}")

    plt.tight_layout()
    plt.show()

    # # Visualize xyz_source, xyz_target
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
