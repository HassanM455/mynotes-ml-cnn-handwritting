import unittest
from unittest.mock import MagicMock
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from src.fs.loader import LetterImageLoader  
from src.strategy.image.normalization import PytorchTensorImageMeanStrategy
from src.strategy.image.commons import ImageContext
import numpy as np

class TestPytorchTensorImageMeanStrategy(unittest.TestCase):

    def setUp(self):
        # Create a mock LetterImageLoader
        self.mock_loader = MagicMock(spec=LetterImageLoader)

        def create_activate_image_generator(num_images, mode='RGB', letter=None):
            mode = mode
            def generator():
                for i in range(num_images):
                    if mode == 'RGB':
                        color = (i * 10, i * 10, i * 10)
                    elif mode == 'L':
                        color = (i*10)
                    yield Image.new(mode, (10, 10), color=color)
            return generator

        self.imode_image_generator = create_activate_image_generator


    def test_rgb_mean_tensor_valid(self): 
        """
        Test that the mean tensor calculation for RGB images returns a tensor of length 3.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        calc_result = next(mean_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        
        # verify its a vector of length of length 3 - RGB
        self.assertEqual(calc_result.shape[0], 3)

    def test_grayscale_mean_tensor_valid(self): 
        """
        Test that the mean tensor calculation for grayscale images returns a scalar or a tensor of length 1.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='L')
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        calc_result = next(mean_strategy(icontext))

        # verify that its essentially a scalar, or a vector of length 1 - Grayscale
        self.assertIs(type(calc_result), torch.Tensor)
        self.assertEqual(calc_result.numel(), 1)

    def test_rgb_mean_tensor_valid_batch(self): 
        """
        Test that the mean tensor calculation for RGB images with batching returns a tensor of length 3.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 3)
        calc_result = next(mean_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        
        # verify its a vector of length of length 3 - RGB
        self.assertEqual(calc_result.shape[0], 3)

    def test_grayscale_mean_tensor_valid_batchsize(self): 
        """
        Test that the mean tensor calculation for grayscale images with batching returns a scalar or a tensor of length 1.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='L')
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 2)
        calc_result = next(mean_strategy(icontext))

        # verify that its essentially a scalar, or a vector of length 1 - Grayscale
        self.assertIs(type(calc_result), torch.Tensor)
        self.assertEqual(calc_result.numel(), 1)


    def test_rgb_mean_tensor_valid_batch_vs_all(self): 
        """
        Test that the mean tensor calculation for RGB images with and without batching returns the same result.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        all_calc_result = next(mean_strategy(icontext))

        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 3)
        batch_calc_result = next(mean_strategy(icontext))

        self.assertTrue(torch.allclose(all_calc_result, batch_calc_result, rtol=1e-05, atol=1e-08))

    def test_rgb_mean_tensor_calculation_values(self):
        """
        Test that the mean tensor calculation for RGB images matches manually calculated values.
        """

        def create_image_generator_rgb():
            for _ in range(3):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(10, 10, 10), dtype=np.uint8))

        def create_image_generator_rgb2():
            for _ in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(10, 10, 10), dtype=np.uint8))



        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        all_calc_result = next(mean_strategy(icontext))

        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb2()]
            )
        )
        sum_by_example = torch.sum(examples, axis=0)
        sum_by_channel = torch.sum(sum_by_example, axis=(0,1))
        W, H = examples.shape[1], examples.shape[2]
        mean_result_from_manual = sum_by_channel / (examples.shape[0]*W*H)
        self.assertTrue(torch.allclose(mean_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))
   
    def test_rgb_mean_tensor_calculation_values_failure(self):
        """
        Test that the mean tensor calculation for RGB images does not match incorrect manually calculated values.
        """

        def create_image_generator_rgb():
            for i in range(3):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i * 10, i*10, i*10), dtype=np.uint8))

        def create_image_generator_rgb2():
            for i in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i * 10, i*10, i*10), dtype=np.uint8))



        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        all_calc_result = next(mean_strategy(icontext))

        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb2()]
            )
        )
        sum_by_example = torch.sum(examples, axis=0)
        sum_by_channel = torch.sum(sum_by_example, axis=(0,1))
        W, H = examples.shape[1], examples.shape[2]
        mean_result_from_manual = sum_by_channel / (examples.shape[0]*W*H)
        self.assertFalse(torch.allclose(mean_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))
   

    def test_grayscale_mean_tensor_calculation_values(self):
        """
        Test that the mean tensor calculation for grayscale images matches manually calculated values.
        """

        def create_image_generator_grayscale():
            for _ in range(3):
                yield Image.fromarray(np.full((2, 2), fill_value=10, dtype=np.uint8))

        def create_image_generator_grayscale2():
            for _ in range(5):
                yield Image.fromarray(np.full((2, 2), fill_value=10, dtype=np.uint8))



        self.mock_loader.gen_all_images.side_effect = create_image_generator_grayscale
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 0)
        all_calc_result = next(mean_strategy(icontext))

        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_grayscale2()]
            )
        )
        sum_by_example = torch.sum(examples, axis=0)
        sum_by_channel = torch.sum(sum_by_example, axis=(0,1))
        W, H = examples.shape[1], examples.shape[2]
        mean_result_from_manual = sum_by_channel / (examples.shape[0]*W*H)
        self.assertTrue(torch.allclose(mean_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))


    def test_rgb_mean_tensor_calculation_values_batch(self):
        """
        Test that the mean tensor calculation for RGB images with batching matches manually calculated values.
        """

        def create_image_generator_rgb():
            for _ in range(9):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(10, 10, 10), dtype=np.uint8))

        def create_image_generator_rgb2():
            for _ in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(10, 10, 10), dtype=np.uint8))



        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        mean_strategy = PytorchTensorImageMeanStrategy(batch_size = 2)
        all_calc_result = next(mean_strategy(icontext))

        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb2()]
            )
        )
        sum_by_example = torch.sum(examples, axis=0)
        sum_by_channel = torch.sum(sum_by_example, axis=(0,1))
        W, H = examples.shape[1], examples.shape[2]
        mean_result_from_manual = sum_by_channel / (examples.shape[0]*W*H)
        self.assertTrue(torch.allclose(mean_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))



if __name__ == '__main__':
    unittest.main()

