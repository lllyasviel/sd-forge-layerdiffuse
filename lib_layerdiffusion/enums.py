from enum import Enum


class ResizeMode(Enum):
    RESIZE = "Just Resize"
    CROP_AND_RESIZE = "Crop and Resize"
    RESIZE_AND_FILL = "Resize and Fill"

    def int_value(self):
        if self == ResizeMode.RESIZE:
            return 0
        elif self == ResizeMode.CROP_AND_RESIZE:
            return 1
        elif self == ResizeMode.RESIZE_AND_FILL:
            return 2
        return 0
