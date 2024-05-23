import io
import asyncio
import aiofiles
import numpy as np
from typing import (
    Union, Sequence, 
    Optional, Tuple,
    Generator, AsyncGenerator,
    List, 
)
from PIL import Image
from collections.abc import Iterable




##################### OLD STUFF ####################3

def get_centered_rectangle(
    image: Image.Image, 
    crop_box_dim: Optional[Tuple[int, int]] = None
) -> Tuple[int, int, int, int]:
    
    mid_img_width = image.width / 2
    mid_img_height = image.height / 2

    quarter_img_width = image.width / 4
    quarter_img_height = image.height / 4

    if not crop_box_dim: 
        left = int(mid_img_width - quarter_img_width)
        upper = int(mid_img_height - quarter_img_height)
        right = int(mid_img_width + quarter_img_width)
        lower = int(mid_img_height + quarter_img_height)
    else:
        crop_width, crop_height = crop_box_dim
        left = int(mid_img_width - crop_width / 2)
        upper = int(mid_img_height - crop_height / 2)
        right = int(mid_img_width + crop_width / 2)
        lower = int(mid_img_height + crop_height / 2)
    return (left, upper, right, lower)


def crop_image_center(
    image: Image.Image,
    crop_box_dim: Optional[Tuple[int, int]] = None
) -> Image.Image:

    centered_rectangle = get_centered_rectangle(
        image,
        crop_box_dim
    )
    
    return image.crop(centered_rectangle)



def crop_image(
    image: Image.Image,
    left: int, 
    upper: int, 
    right: int, 
    lower: int,
) -> Image.Image:

    box = (left, upper, right, lower)
    return image.crop(box)


def convert_img_to_nparr(
    images: Union[Image.Image, Sequence[Image.Image]], 
    ret_as_seq: Optional[bool] = None
) -> np.array:

    if isinstance(images, Iterable):
        np_arrays = [np.asarray(image) for image in images] 
    else: 
        np.asarray(images)
    
    if ret_as_seq is not None:
        np_arrays = np.stack(np_arrays, axis=0)
    
    return np_arrays


async def async_show_images(img_gen: AsyncGenerator[Image.Image, None]): 
    
    async for image in img_gen:
        image.show()
        await asyncio.to_thread(input, "Press Enter to see the next image...")



def show_images(img_gen: Generator[Image.Image, None, None]): 
    
    for image in img_gen:
        image.show()
        input("Press Enter to see the next image...")


def get_image_tiles(
    img: Image.Image,
    patch_height: int,
    patch_width: int
) -> Sequence[Image.Image]:
    img_width, img_height = img.size
    
    if img_width % patch_width != 0 or img_height % patch_height != 0:
        raise Exception(
            f"Image size {img.size} cannot produce equal size patches "
            f"of height, width {(patch_height, patch_width)}"
        )

    patches = []
    height_last_pixel = img_height - patch_height + 1 
    width_last_pixel = img_width - patch_width + 1

    for h in range(0, height_last_pixel, patch_height):
        for w in range(0, width_last_pixel, patch_width):
            patch = crop_image(
                img,
                *(w, h, w + patch_width, h + patch_height)
            )
            patches.append(patch)

    return patches


def get_image_patches(
    img: Image.Image,
    patch_height: int,
    patch_width: int, 
    stride_height: int = 1,
    stride_width: int = 1
) -> Sequence[Image.Image]:
    
    img_width, img_height = img.size
    
    patches = []
    height_last_pixel = img_height - patch_height + 1 
    width_last_pixel = img_width - patch_width + 1
    
    for h in range(0, height_last_pixel, stride_height):
        for w in range(0, width_last_pixel, stride_width):
            patch = crop_image(
                img,
                *(w, h, w + patch_width, h + patch_height)
            )
            patches.append(patch)

    return patches


def get_image_vectors(
    images: Union[Image.Image, List[Image.Image]]
) -> List[np.array]:
    '''
    PIL image shape = (width, height, channel) 
    PIL.Image.Image -> numpy.ndarray
    numpy.ndarray shape = (height, width, channel)
    3d-array -> 1d-array = (height x width x channel)

    1d-array layout = [h1w1c1, h1w1c2, h1w1c3, h1w2c1, h1w2c2, h1w2c3, ...]
    implying "hold height constant, hold width constant, iterate over channels, 
    "hold height constant, iterate to next width index, iterate over channels ..."
    '''

    images = images if '__iter__' in dir(images) else [images]
    
    return [np.array(image).flatten() for image in images]




