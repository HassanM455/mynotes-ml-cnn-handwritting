import unittest
from unittest.mock import MagicMock
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from src.fs.loader import LetterImageLoader 
from src.strategy.image.normalization import PytorchTensorImageNormalizationStrategy
from src.strategy.image.commons import ImageContext
import numpy as np

class TestPytorchTensorImageNormalizationStrategy(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment by creating a mock LetterImageLoader and a function to generate images.
        """
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

    def test_rgb_normalization_tensor_valid(self):
        """
        Test the normalization strategy with RGB images ensuring the output is a tensor with the correct shape.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        calc_result = next(normalization_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        
        # Verify that 10 examples are returned.
        self.assertEqual(calc_result.shape[0], 10)

        # Verify that the shape of the returned tensor is as expected.
        self.assertEqual(list(calc_result.shape), [10, 10, 10, 3])

    def test_grayscale_normalization_tensor_valid(self): 
        """
        Test the normalization strategy with grayscale images ensuring the output is a tensor with the correct shape.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='L')
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        calc_result = next(normalization_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        self.assertEqual(calc_result.shape[0], 10)

        # Verify that the shape of the returned tensor is as expected.
        self.assertEqual(list(calc_result.shape), [10, 10, 10])

    def test_rgb_normalization_tensor_valid_batch(self):
        """
        Test the normalization strategy with RGB images and a batch size ensuring the output tensor has the 
        correct shape.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=3)
        calc_result = next(normalization_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        
        # Verify that the shape of the batch is as expected with batch size 3.
        self.assertEqual(list(calc_result.shape), [3, 10, 10, 3])

    def test_grayscale_normalization_tensor_valid_batchsize(self):
        """
        Test the normalization strategy with grayscale images and a batch size ensuring the output tensor has 
        the correct shape.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='L')
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=2)
        calc_result = next(normalization_strategy(icontext))

        self.assertIs(type(calc_result), torch.Tensor)
        self.assertEqual(list(calc_result.shape), [2, 10, 10])

    def test_rgb_normalization_tensor_valid_batch_vs_all(self):
        """
        Test the normalization strategy with RGB images ensuring that the result of normalizing all at once is 
        the same as normalizing in batches.
        """
        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        all_calc_result = next(normalization_strategy(icontext))

        self.mock_loader.gen_all_images.side_effect = self.imode_image_generator(10, mode='RGB')
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=3)
        tensors = [tensor for tensor in normalization_strategy(icontext)]
        batch_calc_result = torch.cat(tensors, dim=0)        

        self.assertTrue(torch.allclose(all_calc_result, batch_calc_result, rtol=1e-05, atol=1e-08))

    def test_rgb_normalization_tensor_calculation_values(self):
        """
        Test the normalization strategy with RGB images ensuring that the calculated values are correct 
        by comparing with manually computed values.
        """
        def create_image_generator_rgb():
            for i in range(3):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i*10, i*10, i*10), dtype=np.uint8))

        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        all_calc_result = next(normalization_strategy(icontext))
        mean_tensor = icontext.mean_tensor
        std_tensor = icontext.std_tensor

        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb()]
            )
        )

        normalization_result_from_manual = (examples - mean_tensor) / std_tensor 
        self.assertTrue(torch.allclose(normalization_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))

    def test_rgb_normalization_tensor_calculation_values_failure(self):
        """
        Test the normalization strategy with RGB images ensuring that the calculation values are incorrect 
        when the generated image values differ.
        """
        def create_image_generator_rgb():
            for i in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i * 10 + 1, i*10, i*10), dtype=np.uint8))

        def create_image_generator_rgb2():
            for i in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i * 10, i*10, i*10), dtype=np.uint8))

        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        all_calc_result = next(normalization_strategy(icontext))
        mean_tensor = icontext.mean_tensor
        std_tensor = icontext.std_tensor

        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb2()]
            )
        )
        normalization_result_from_manual = (examples - mean_tensor) / std_tensor 
        self.assertFalse(torch.allclose(normalization_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))

    def test_grayscale_normalization_tensor_calculation_values(self):
        """
        Test the normalization strategy with grayscale images ensuring that the calculated values are correct 
        by comparing with manually computed values.
        """
        def create_image_generator_grayscale():
            for i in range(5):
                yield Image.fromarray(np.full((2, 2), fill_value=i * 10, dtype=np.uint8))

        self.mock_loader.gen_all_images.side_effect = create_image_generator_grayscale
        icontext = ImageContext(image_mode='grayscale', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=0)
        all_calc_result = next(normalization_strategy(icontext))
        mean_tensor = icontext.mean_tensor
        std_tensor = icontext.std_tensor
        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_grayscale()]
            )
        )

        normalization_result_from_manual = (examples - mean_tensor) / std_tensor 
        self.assertTrue(torch.allclose(normalization_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))

    def test_rgb_normalization_tensor_calculation_values_batch(self):
        """
        Test the calculation of standard deviation tensor for RGB images with batching by comparing the 
        result with a manually computed value.
        """
        def create_image_generator_rgb():
            for i in range(9):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i*10, i*10, i*10), dtype=np.uint8))

        def create_image_generator_rgb2():
            for i in range(5):
                yield Image.fromarray(np.full((2, 2, 3), fill_value=(i*10, i*10, i*10), dtype=np.uint8))

        batch_size = 2
        self.mock_loader.gen_all_images.side_effect = create_image_generator_rgb
        icontext = ImageContext(image_mode='rgb', image_loader=self.mock_loader)
        normalization_strategy = PytorchTensorImageNormalizationStrategy(batch_size=batch_size)
        all_calc_result = next(normalization_strategy(icontext))
        mean_tensor = icontext.mean_tensor
        std_tensor = icontext.std_tensor
        
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb2()][0 : batch_size]
            )
        )

        normalization_result_from_manual = (examples - mean_tensor) / std_tensor
        print(normalization_result_from_manual[0] - examples[0])
        self.assertTrue(torch.allclose(normalization_result_from_manual, all_calc_result, rtol=1e-04, atol=1e-05))

if __name__ == '__main__':
    unittest.main()

