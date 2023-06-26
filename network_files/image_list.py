from typing import List, Tuple
from torch import Tensor

"""
    包含图像列表的结构（可能是大小不等）作为一个张量。
    这是通过将图像填充到相同的大小来实现的，并将每个图像的原始大小存储在字段中
"""
class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """
    """
        Arguments:
            tensors (tensor) padding后的图像数据
            image_sizes (list[tuple[int, int]])  padding前的图像尺寸
    """
    def __init__(self, tensors, image_sizes):
        # type: (Tensor, List[Tuple[int, int]]) -> None

        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        # type: (Device) -> ImageList # noqa
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)

