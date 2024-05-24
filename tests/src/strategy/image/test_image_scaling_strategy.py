import unittest
from unittest.case import _AssertRaisesContext
from unittest.mock import MagicMock
from src.strategy import image
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from src.fs.loader import LetterImageLoader  
from src.strategy.image.scaling import PILImageScalingStrategy
from src.strategy.image.commons import ImageContext, ImagePreProcScalingContext
import numpy as np


def create_image_gen(mode='RGB', size=(10,20), letter = None):
    def image_generator():
        if mode == 'RGB':
            color = (10, 10, 10)
        else:
            color = (10)
        for _ in range(3):
            image = Image.new(mode, size, color = color)
            if letter == None:
                yield image                    
            else:
                yield (image, letter)

    return image_generator

def create_image_and_letter_gen(mode='RGB', size=(10,20)):
    def image_generator():
        if mode == 'RGB':
            color = (10, 10, 10)
        else:
            color = (10)
        count = 0
        for _ in range(1):
            image = Image.new(mode, size, color = color)

            if count < 2:
                letter = 'A'
            else:
                letter = 'B'
            yield (image, letter)

    return image_generator


class TestPILImageScalingStrategy(unittest.TestCase):

    def setUp(self):

        mock_loader = MagicMock(spec=LetterImageLoader)
        mock_loader.gen_all_images = create_image_gen()
        mock_loader.gen_all_images_by_letter = create_image_and_letter_gen()
        self.mock_loader = mock_loader

        self.icontext = ImageContext(mock_loader, image_mode = 'rgb')
        self.scontext = ImagePreProcScalingContext(5,10)
        self.scontext_square = ImagePreProcScalingContext(5,10, make_square=True)
    
    def test_scale_images(self):
        scaling_strategy = PILImageScalingStrategy(batch_size=0)
        input_images = [image for image in self.mock_loader.gen_all_images()]
        size = input_images[0].size
        print(f'Input size: {size}')
        
        scaled_img_gen = scaling_strategy(self.icontext, self.scontext)
        scaled_imgs = next(scaled_img_gen)

        self.assertEqual((5,10), scaled_imgs[0].size)

    def test_scale_images_square(self):
        scaling_strategy = PILImageScalingStrategy(batch_size=0)
        input_images = [image for image in self.mock_loader.gen_all_images()]
        size = input_images[0].size
        print(f'Input size: {size}')
        
        scaled_img_gen = scaling_strategy(self.icontext, self.scontext_square)
        scaled_imgs = next(scaled_img_gen)

        self.assertEqual((10,10), scaled_imgs[0].size)

    def test_init_scontext_invalid_type(self):
        scaling_strategy = PILImageScalingStrategy(batch_size=0)

        with self.assertRaises(Exception) as ec:
            next(scaling_strategy(object(), self.icontext))
        self.assertIn("Argument `icontext` must be of type", str(ec.exception)) 

        with self.assertRaises(Exception) as ec:
            next(scaling_strategy(self.icontext, object()))
        self.assertIn("Argument `scontext` must be of type", str(ec.exception)) 




if __name__ == '__main__':
    unittest.main()
