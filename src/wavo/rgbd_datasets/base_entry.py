"""Abstract base entry for RGBD datasets."""

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, computed_field

from wavo.image import RGBDImage


class BaseRGBDEntry(BaseModel, ABC):
    """A single RGBD dataset entry: an RGB/depth image pair with their timestamps.

    Subclasses implement :meth:`load` with the dataset-specific reading logic
    (file formats, depth scaling, timestamp parsing, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgbd_image: RGBDImage
    rgb_stamp: float
    depth_stamp: float

    @computed_field
    @property
    def stamp(self) -> float:
        """Average of the RGB and depth timestamps."""
        return 0.5 * (self.rgb_stamp + self.depth_stamp)

    @classmethod
    @abstractmethod
    def load(cls, *args, **kwargs) -> "BaseRGBDEntry":
        """Read a single entry from disk and return it as a model instance."""
        raise NotImplementedError
