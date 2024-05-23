import torch
import string
import numpy as np
from typing import (
    Tuple, Generator,
    Optional
)
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngImageFile
from src.fs.loader import LetterImageLoader

class ImageContext:
    def __init__(
        self,
        image_loader: LetterImageLoader,
        image_mode: str,
        reshape: Optional[Tuple[int, int]] = None
    ):
        
        if image_mode not in ['rgb', 'grayscale']:
            raise Exception(
                f"Unknown `image_mode` parameter with value: {image_mode}. " 
                f"Allowed values are ['rgb', 'grayscale'] "
            )

        if not isinstance(image_loader, LetterImageLoader):
            raise Exception(
                f'Value passed to argument `image_loader` must be of type {LetterImageLoader}. The value passed '
                f'is of type: {type(image_loader)}).'
            )


        self.ildr = image_loader
        self.imode = image_mode
        self.num_of_examples: int = 3
        self._mean_tensor: torch.Tensor = None
        self._std_tensor: torch.Tensor = None
        self._run_context_validation()

    @property
    def mean_tensor(self) -> torch.Tensor:
        if self._mean_tensor is None: 
            raise AttributeError(f'Object property `mean_tensor` has not yet been set. ')
        return self._mean_tensor

    @mean_tensor.setter
    def mean_tensor(self, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise Exception(f"Property `mean_tensor` must be of type `{torch.Tensor}`. Provided type: {type(value)}.")
        self._mean_tensor = value
    
    @property
    def std_tensor(self) -> torch.Tensor:
        if self._std_tensor is None:
            raise AttributeError(f'Object property `std_tensor` has not yet been set. ')
        return self._std_tensor

    @std_tensor.setter
    def std_tensor(self, value):
        if not isinstance(value, torch.Tensor):
            raise Exception(f"Property `std_tensor` must be of type `{torch.Tensor}`. Provided type: {type(value)}.")
        self._std_tensor = value

    def _count_images(self) -> None: 
        data_gen = self.ildr.gen_all_images()
        cnt = 0
        for _ in data_gen: 
            cnt += 1    
        self.num_of_examples = cnt
        return None

    def _run_context_validation(self) -> None:
        self._check_all_pil_images_objects()
        self._count_images()
        self._check_input_image_mode()
  
    def _check_all_pil_images_objects(self, ) -> None:
        image_gen  = self.ildr.gen_all_images()
        if all(isinstance(img, PngImageFile) or isinstance(img, Image.Image) for img in image_gen):
            return None
        raise Exception(
            f"Some or all images loaded by `LetterImageLoader` object are not of types: "
            f"{PngImageFile} or {Image.Image}."
        )

    def _check_input_image_mode(self) -> None: 
        expected_imode = self.imode
        mode_trans = {
            "l": "grayscale",
            "rgb": "rgb"
        }

        for image in self.ildr.gen_all_images():
            actual_imode = image.mode
            trans_actual_imode = mode_trans[actual_imode.lower()]
            if expected_imode != trans_actual_imode :
                raise Exception(
                    f"Expected `image_mode` equal to `{expected_imode}`, but input image has mode '{image.mode}'. "
                )

        return None


        
class ImageMixin:

    @staticmethod
    def _gen_img_batch_as_nparray(
        icontext: ImageContext,
        batch_size: int       
    ) -> Generator[np.array, None, None]:

        batch = []
        img_gen = icontext.ildr.gen_all_images()
        for img in img_gen :
            np_array = np.array(img)
            batch.append(np_array)
            if batch_size == len(batch): 
                yield np.array(batch)
                batch = []

        if not batch:
            raise Exception(
                f"The batch is empty when generator `{icontext.ildr.gen_all_images}` is called. "
            )
        
        yield np.array(batch)

    @staticmethod
    def _gen_img_letter_batch_as_nparray(
        icontext: ImageContext,
        batch_size: int, 
        letter: str
    ) -> Generator[np.array, None, None]:

        if letter not in (string.ascii_uppercase + string.ascii_lowercase):
            raise Exception(
                f"Unknown letter type passed to argument `letter` with value: {letter}. 'Letter' must be one of \n"
                f"the lower or upper case ascii characters as defined in the `string` library. "
            )

        batch = []
        letter_img_gen = icontext.ildr.gen_letter_images(letter)
        for img in letter_img_gen :
            np_array = np.array(img)
            batch.append(np_array)
            if batch_size == len(batch):
                yield np.array(batch)
                batch = []

        if not batch:
            raise Exception(
                f"The batch is empty when generator `{icontext.ildr.gen_letter_images}` is called. "
            )
        
        yield np.array(batch)



