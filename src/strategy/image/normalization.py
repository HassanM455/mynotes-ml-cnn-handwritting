import torch
import string
import numpy as np
from typing import (
    Generator, 
)
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngImageFile
from .commons import ImageContext, ImageMixin


class PytorchTensorImageMeanStrategy(ImageMixin):
    def __init__(
        self,
        batch_size: int = 0
    ):
        self.batch_size = batch_size 

    def __call__(
        self, 
        icontext: ImageContext
    ) -> Generator[torch.Tensor, None, None]:
    
        if not isinstance(icontext, ImageContext):
            raise Exception(f"Argument `context` must be of type {ImageContext}.")
        mean_tensor = self._get_mean_tensor(icontext)
        icontext.mean_tensor = mean_tensor
        yield mean_tensor

    def _get_mean_tensor(
        self,
        icontext: ImageContext
    ) -> torch.Tensor:

        image_mode = icontext.imode
        if image_mode == 'grayscale' or image_mode == 'rgb':
            return self._handle_rgb_and_grayscale_mean_calc(icontext)

        raise Exception(f"Unknown `context.image_mode` value: {image_mode}.")

    def _handle_rgb_and_grayscale_mean_calc(self, icontext: ImageContext) -> torch.Tensor:
       
        data_gen = self._gen_img_batch_as_nparray(icontext, self.batch_size)
        
        first_batch = next(data_gen) 
        tensor_batch = torch.tensor(first_batch, dtype=torch.float32)

        # Formula : mean_new = (mean_prev)*prev_batchsize + (cur_batch_mean)*cur_batch_size
        #                      -------------------------------------------------------------
        #                                       prev_batchsize + cur_batch_size        
 
        sum_tensor = torch.sum(tensor_batch, dim=0)
        num_of_examples = tensor_batch.shape[0]
        
        while True:
            if num_of_examples >= icontext.num_of_examples:
                break

            cur_batch = next(data_gen)
            cur_tensor = torch.tensor(cur_batch, dtype=torch.float32)
            sum_tensor += torch.sum(cur_tensor, dim=0)
            num_of_examples += cur_tensor.shape[0]
        
        sum_tensor_shape = sum_tensor.shape
        denominator = num_of_examples*sum_tensor_shape[0]*sum_tensor_shape[1]
        sum_rgb_tensor = torch.sum(sum_tensor, axis = (0,1))
        rgb_mean_tensor = sum_rgb_tensor / denominator
 
        return rgb_mean_tensor


class PytorchTensorImageStdStrategy(ImageMixin):
    def __init__(
        self,
        batch_size: int = 0
    ):
        self.batch_size = batch_size 

    def __call__(
        self, 
        icontext: ImageContext,
    ) -> Generator[torch.Tensor, None, None]:

        if not isinstance(icontext, ImageContext):
            raise Exception(f"Argument `context` must be of type {ImageContext}.")

        try:
            mean_tensor = icontext.mean_tensor
        except AttributeError as e:
            mean_strategy = PytorchTensorImageMeanStrategy(self.batch_size)
            mean_tensor = next(mean_strategy(icontext))
            icontext.mean_tensor = mean_tensor

        std_tensor = self._get_std_tensor(icontext)
        icontext.std_tensor = std_tensor
        yield std_tensor



    def _get_std_tensor(
        self,
        icontext: ImageContext,
    ) -> torch.Tensor:

        image_mode = icontext.imode
        if image_mode == 'grayscale' or image_mode == 'rgb':
            return self._handle_rgb_and_grayscale_std_calc(icontext)
        
        raise Exception(f"Unknown `context.image_mode` value: {image_mode}.")

    def _handle_rgb_and_grayscale_std_calc(self, icontext: ImageContext) -> torch.Tensor:
        
        mean_tensor = icontext.mean_tensor
        data_gen = self._gen_img_batch_as_nparray(icontext, self.batch_size)
        
        first_batch = next(data_gen) 
        tensor_batch = torch.tensor(first_batch, dtype=torch.float32)

        sum_squared_diff = torch.sum((tensor_batch - mean_tensor) ** 2, dim=[0, 1, 2])
        num_of_examples = tensor_batch.shape[0]

        while True:
            if num_of_examples >= icontext.num_of_examples:
                break

            cur_batch = next(data_gen)
            tensor = torch.tensor(cur_batch, dtype=torch.float32)
            batch_size = tensor.shape[0]
            batch_squared_diff = torch.sum((tensor - mean_tensor) ** 2, dim=[0, 1, 2])

            sum_squared_diff += batch_squared_diff
            num_of_examples += batch_size

        
        W, H = first_batch.shape[1], first_batch.shape[2]
        total_pixels = num_of_examples*W*H
        variance_tensor = sum_squared_diff / total_pixels
        std_tensor = torch.sqrt(variance_tensor)
        
        return std_tensor
    



class PytorchTensorImageNormalizationStrategy(ImageMixin):
    def __init__(
        self,
        batch_size: int = 0
    ):
        self.batch_size = batch_size 

    def __call__(
        self, 
        icontext: ImageContext
    ) -> Generator[torch.Tensor, None, None]: 
    
        if not isinstance(icontext, ImageContext):
            raise Exception(f"Argument `context` must be of type {ImageContext}.")

        try:
            mean_tensor = icontext.mean_tensor
        except AttributeError as e:
            mean_strategy = PytorchTensorImageMeanStrategy(self.batch_size)
            mean_tensor = next(mean_strategy(icontext))
            icontext.mean_tensor = mean_tensor

        try:
            std_tensor = icontext.std_tensor
        except AttributeError as e:
            std_strategy = PytorchTensorImageStdStrategy(self.batch_size)
            std_tensor = next(std_strategy(icontext))
            icontext.std_tensor = std_tensor


        tensor_gen = self._get_normalized_tensor(icontext)
        
        for tensor in tensor_gen: 
            yield tensor


    def _get_normalized_tensor(self, icontext: ImageContext) -> Generator[torch.Tensor, None, None]: 

        image_mode = icontext.imode
        if not image_mode == 'grayscale' and not image_mode == 'rgb':
            raise Exception(f"Unknown `context.immode` value: {image_mode}.")

        tensor_gen = self._handle_rgb_and_grayscale_norm_calc(icontext)
        for tensor in tensor_gen:
            yield tensor

    def _handle_rgb_and_grayscale_norm_calc(self, icontext: ImageContext) -> Generator[torch.Tensor, None, None] :
        
        mean_tensor = icontext.mean_tensor
        std_tensor = icontext.std_tensor
        data_gen = self._gen_img_batch_as_nparray(icontext, self.batch_size)
        
        first_batch = next(data_gen) 
        tensor_batch = torch.tensor(first_batch, dtype=torch.float32)

        normalized_tensor = (tensor_batch - mean_tensor)/std_tensor
        num_of_examples = normalized_tensor.shape[0]

        yield normalized_tensor

        while True:
            if num_of_examples >= icontext.num_of_examples:
                break

            cur_batch = next(data_gen)
            tensor = torch.tensor(cur_batch, dtype=torch.float32)
            batch_size = tensor.shape[0]
            normalized_tensor = (tensor - mean_tensor)/std_tensor
            num_of_examples += batch_size

            yield normalized_tensor



