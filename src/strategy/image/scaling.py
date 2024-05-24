from .commons import (
    ImageContext, ImageMixin,
    ImagePreProcScalingContext
)
from src.fs.loader import LetterImageLoader
from PIL import Image
from typing import (
    Generator, Tuple
)



class PILImageScalingStrategy(ImageMixin):

    def __init__(self, batch_size: int): 
        self.batch_size = batch_size

    def __call__(
        self, 
        icontext: ImageContext,
        scontext: ImagePreProcScalingContext
    ) -> Generator[Tuple[Image.Image,...], None, None]:
        
        if not isinstance(icontext, ImageContext):
            raise Exception(f"Argument `icontext` must be of type {ImageContext}.")

        if not isinstance(scontext, ImagePreProcScalingContext):
            raise Exception(f"Argument `scontext` must be of type {ImagePreProcScalingContext}.")

        scaled_image_gen = self._scale_images(icontext, scontext)

        for scaled_image in scaled_image_gen:
            yield scaled_image

    def _scale_images(
        self, 
        icontext: ImageContext,
        scontext: ImagePreProcScalingContext
    ) -> Generator[Tuple[Image.Image,...], None, None]:
        image_batch_gen = self._gen_img_batch(icontext, self.batch_size)
        
        for image_batch in image_batch_gen:
            rescaled_images = self._rescale_strategy(image_batch, scontext)
            if scontext.make_square:                
                rescaled_images = self._get_squared_images(rescaled_images)
            yield rescaled_images


    def _rescale_strategy(
        self,
        images: Tuple[Image.Image, ...],
        scontext: ImagePreProcScalingContext
    ) -> Tuple[Image.Image, ...]:

        target_size = scontext.new_scale_size
        scaled_images = []
        for image in images:
            image.thumbnail(target_size, Image.LANCZOS)
            scaled_images.append(image)

        return tuple(scaled_images)





