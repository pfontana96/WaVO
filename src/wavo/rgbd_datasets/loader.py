"""Generic loader interface for RGBD datasets."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterator

from wavo.rgbd_datasets.base_entry import BaseRGBDEntry

from wavo._core.image import CameraIntrinsics


class RGBDDatasetLoader(ABC):
    """Indexable/iterable loader over the frames of an RGBD dataset.

    Subclasses implement the dataset-specific parsing and register themselves
    for a format name at class-creation time::

        class TUMRGBDDatasetLoader(RGBDDatasetLoader, fmt="tum"):
            ...

    Consumers then obtain the right loader through the base class::

        dataset = RGBDDatasetLoader.load("tum", root, intrinsics=...)

    Entries are loaded lazily: ``__getitem__``/``__iter__`` read images from
    disk on demand instead of holding the whole sequence in memory.
    """

    _registry: ClassVar[dict[str, type["RGBDDatasetLoader"]]] = {}

    fmt: ClassVar[str | None] = None
    """Format name this loader is registered under (``None`` for the base)."""

    def __init_subclass__(cls, fmt: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if fmt is not None:
            registered = RGBDDatasetLoader._registry.get(fmt)
            if registered is not None and registered is not cls:
                raise ValueError(
                    f"format {fmt!r} is already registered to {registered.__name__}"
                )
            cls.fmt = fmt
            RGBDDatasetLoader._registry[fmt] = cls

    def __init__(self, root: Path, intrinsics: CameraIntrinsics | None = None):
        self.root = Path(root)
        self.intrinsics = intrinsics

    @classmethod
    def load(cls, fmt: str, path: Path, **kwargs) -> "RGBDDatasetLoader":
        """Instantiate the loader registered for ``fmt`` on the dataset at ``path``.

        Extra keyword arguments are forwarded to the loader's constructor
        (e.g. ``intrinsics=...`` for formats that don't ship them).
        """
        try:
            loader_cls = cls._registry[fmt]
        except KeyError:
            known = ", ".join(sorted(cls._registry)) or "none"
            raise ValueError(
                f"unsupported dataset format {fmt!r} (registered: {known})"
            ) from None
        return loader_cls(Path(path), **kwargs)

    @abstractmethod
    def __len__(self) -> int:
        """Number of associated RGB/depth pairs in the dataset."""

    @abstractmethod
    def __getitem__(self, i: int) -> BaseRGBDEntry:
        """Read pair ``i`` from disk and return it as an entry."""

    def __iter__(self) -> Iterator[BaseRGBDEntry]:
        for i in range(len(self)):
            yield self[i]
