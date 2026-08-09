"""Loaders for RGBD datasets."""

from pathlib import Path
from typing import Iterator, List, Tuple, Sequence
from time import perf_counter

import cv2
import numpy as np

from .base_entry import BaseRGBDEntry
from wavo.image import RGBDImage


class TUMEntry(BaseRGBDEntry):
    @classmethod
    def load(
        cls,
        root: Path,
        rgb_stamp: float,
        rgb_file: str,
        depth_stamp: float,
        depth_file: str,
        depth_scale: float = 5000.0,
    ) -> "TUMEntry":

        bgr = cv2.imread(str(root / rgb_file))
        depth = (
            cv2.imread(str(root / depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32)
            / depth_scale
        )

        image = RGBDImage(bgr, depth)

        return cls(
            rgbd_image=image,
            rgb_stamp=rgb_stamp,
            depth_stamp=depth_stamp,
        )


class RGBDDatasetLoader:
    """Indexable/iterable loader for an RGBD dataset. Only TUM format for now.

    Pairs each rgb frame with the nearest depth frame in time (within
    ``max_stamp_diff`` seconds); unmatched rgb frames are skipped.
    """

    def __init__(
        self,
        root: Path,
        fmt: str = "tum",
        depth_scale: float = 5000.0,
        max_stamp_diff: float = 0.02,
        time_offset: float = 0.0,
    ):
        if fmt != "tum":
            raise ValueError(f"unsupported dataset format: {fmt!r}")
        self.root = Path(root)
        self.depth_scale = depth_scale
        rgb_index = self._read_index(self.root / "rgb.txt")
        depth_index = self._read_index(self.root / "depth.txt")

        self._n_rgb, self._n_depth = len(rgb_index), len(depth_index)
        self._pairs = self._associate(
            rgb_index, depth_index, max_stamp_diff, time_offset
        )

    @staticmethod
    def _read_index(path: Path) -> List[Tuple[float, str]]:
        entries = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                stamp, filename = line.split()
                entries.append((float(stamp), filename))
        return entries

    @staticmethod
    def _associate(
        rgb: Sequence[Tuple[float, str]],
        depth: Sequence[Tuple[float, str]],
        max_diff: float,
        offset: float = 0.0,
    ) -> List[Tuple[float, str, float, str]]:
        """Greedy mutually-exclusive matching (TUM ``associate.py``).

        Plain per-frame nearest-neighbour matching does NOT retire the depth
        entry it consumed, so one depth frame can end up serving several
        consecutive rgb frames. That is silent and destructive: the depth then
        describes the scene at one instant while the pixels describe another,
        which corrupts any depth-dependent quantity (translation, scale) while
        leaving depth-independent ones (rotation) looking healthy.
        """
        if not rgb or not depth:
            return []

        d_stamps = np.fromiter((d[0] for d in depth), float, len(depth))
        order = np.argsort(d_stamps, kind="stable")
        d_sorted = d_stamps[order] + offset

        # candidate pairs within the window, found by binary search rather than
        # the O(n*m) double loop in the original script
        candidates: List[Tuple[float, int, int]] = []
        for i, (stamp, _) in enumerate(rgb):
            lo = int(np.searchsorted(d_sorted, stamp - max_diff, "left"))
            hi = int(np.searchsorted(d_sorted, stamp + max_diff, "right"))
            for k in range(lo, hi):
                j = int(order[k])
                candidates.append((abs(depth[j][0] + offset - stamp), i, j))

        candidates.sort()

        used_rgb: set = set()
        used_depth: set = set()
        pairs: List[Tuple[float, str, float, str]] = []
        for _, i, j in candidates:
            if i in used_rgb or j in used_depth:
                continue
            used_rgb.add(i)
            used_depth.add(j)
            pairs.append((rgb[i][0], rgb[i][1], depth[j][0], depth[j][1]))

        pairs.sort()
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, i: int) -> TUMEntry:
        rgb_stamp, rgb_file, depth_stamp, depth_file = self._pairs[i]
        start = perf_counter()
        entry = TUMEntry.load(
            self.root, rgb_stamp, rgb_file, depth_stamp, depth_file, self.depth_scale
        )
        end = perf_counter()
        # print(f"Entry creation took: {end - start:.4f} s")
        return entry

    def __iter__(self) -> Iterator[TUMEntry]:
        for i in range(len(self)):
            yield self[i]

    # -- diagnostics ------------------------------------------------------

    @property
    def stamp_offsets(self) -> np.ndarray:
        """|rgb_stamp - depth_stamp| for every pair, in seconds."""
        return np.array([abs(r - d) for r, _, d, _ in self._pairs])

    def summary(self) -> str:
        off = self.stamp_offsets
        if not len(off):
            return "0 pairs associated"
        reused = len(self._pairs) - len({p[3] for p in self._pairs})
        return (
            f"{len(self._pairs)} pairs from {self._n_rgb} rgb / "
            f"{self._n_depth} depth frames "
            f"({self._n_rgb - len(self._pairs)} rgb dropped); "
            f"stamp offset median {np.median(off)*1000:.1f} ms, "
            f"max {off.max()*1000:.1f} ms; reused depth frames: {reused}"
        )
