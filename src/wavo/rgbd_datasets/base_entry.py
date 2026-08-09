"""Entry type for RGBD datasets."""

from pydantic import BaseModel, ConfigDict, computed_field

from wavo._core.image import RGBDFrame


class BaseRGBDEntry(BaseModel):
    """A single RGBD dataset entry: an RGB/depth image pair with their timestamps.

    Loaders build these in ``__getitem__`` with the dataset-specific reading
    logic (file formats, depth scaling, timestamp parsing, ...).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rgbd_frame: RGBDFrame
    rgb_stamp: float
    depth_stamp: float

    @computed_field
    @property
    def stamp(self) -> float:
        """Average of the RGB and depth timestamps."""
        return 0.5 * (self.rgb_stamp + self.depth_stamp)
