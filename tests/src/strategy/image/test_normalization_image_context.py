import unittest
from unittest import mock
from unittest.mock import MagicMock, patch
import torch
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from src.fs.loader import LetterImageLoader 
from src.strategy.image.commons import ImageContext, ImageMixin

class TestImageContext(unittest.TestCase):

    def setUp(self):
        # Create a mock LetterImageLoader
        self.mock_loader = MagicMock(spec=LetterImageLoader)

        # Mock the image generator to return some dummy images
        def image_generator():
            for _ in range(3):
                yield Image.new('RGB', (100, 100))

        self.mock_loader.gen_all_images.side_effect = image_generator

    def test_init_valid_rgb(self):
        """
        Test initializing ImageContext with a valid RGB image mode.

        This test verifies that the ImageContext class can be initialized with a valid
        'rgb' image mode and that the number of examples is correctly set based on
        the number of images provided by the image_loader.

        Assertions:
        - The image mode (imode) is set to 'rgb'.
        - The number of examples (num_of_examples) is set to the number of images
          generated by the image_loader (in this case, 3).
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        self.assertEqual(context.imode, 'rgb')
        self.assertEqual(context.num_of_examples, 3)

    def test_init_valid_grayscale(self):
        """
        Test initializing ImageContext with a valid Grayscale image mode.

        This test verifies that the ImageContext class can be initialized with a valid
        'grayscale' image mode and that the number of examples is correctly set based on
        the number of images provided by the image_loader.

        Assertions:
        - The image mode (imode) is set to 'grayscale'.
        - The number of examples (num_of_examples) is set to the number of images
          generated by the image_loader (in this case, 3).
        """
        def image_generator():
            for _ in range(3):
                yield Image.new('L', (100, 100))
        mock_loader = MagicMock(spec=LetterImageLoader)
        mock_loader.gen_all_images.side_effect = image_generator

        context = ImageContext(image_loader=mock_loader, image_mode='grayscale')
        self.assertEqual(context.imode, 'grayscale')
        self.assertEqual(context.num_of_examples, 3)
    
    def test_invalid_image_mode(self):
        """
        Test initializing ImageContext with an invalid image mode.

        This test verifies that an exception is raised when attempting to initialize 
        the ImageContext class with an invalid image mode. The exception message should 
        indicate that the provided image mode is unknown.

        Assertions:
        - An exception is raised with a message indicating the unknown image mode.
        """
        with self.assertRaises(Exception) as context:
            ImageContext(image_loader=self.mock_loader, image_mode='invalid_mode')
        self.assertIn("Unknown `image_mode` parameter with value: invalid_mode", str(context.exception))

    def test_invalid_image_loader(self):
        """
        Test initializing ImageContext with an invalid image loader.

        This test verifies that an exception is raised when attempting to initialize 
        the ImageContext class with an invalid image loader. The exception message should 
        indicate that the provided image loader must be of type LetterImageLoader.

        Assertions:
        - An exception is raised with a message indicating the invalid type for the image loader.
        """
        with self.assertRaises(Exception) as context:
            ImageContext(image_loader="invalid_loader", image_mode='rgb')
        self.assertIn("Value passed to argument `image_loader` must be of type", str(context.exception))

    def test_mean_tensor_property(self):
        """
        Test setting and getting the mean_tensor property.

        This test verifies the behavior of the mean_tensor property in the ImageContext class.
        It checks that an exception is raised if the mean_tensor property is accessed before 
        it is set. It also verifies that the mean_tensor property can be set and retrieved 
        correctly.

        Assertions:
        - An exception is raised when accessing mean_tensor before it is set.
        - The mean_tensor property can be set to a torch.Tensor value.
        - The mean_tensor property returns the correct tensor value.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        with self.assertRaises(AttributeError) as context_exception:
            _ = context.mean_tensor
        self.assertIn("Object property `mean_tensor` has not yet been set", str(context_exception.exception))
        context.mean_tensor = torch.tensor([0.5, 0.5, 0.5])

        # must use pytorch utility to estimate closeness
        self.assertTrue(torch.equal(context.mean_tensor, torch.tensor([0.5, 0.5, 0.5])))
        
    def test_invalid_mean_tensor_property(self):
        """
        Test setting mean_tensor property with an invalid type.

        This test verifies that an exception is raised when attempting to set the 
        mean_tensor property with a value that is not of type torch.Tensor. The 
        exception message should indicate the required type for the mean_tensor property.

        Assertions:
        - An exception is raised with a message indicating the invalid type for mean_tensor.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        with self.assertRaises(Exception) as context_exception:
            context.mean_tensor = "invalid_tensor"
        self.assertIn("Property `mean_tensor` must be of type `<class 'torch.Tensor'>`", str(context_exception.exception))

    def test_std_tensor_property(self):
        """
        Test setting and getting the std_tensor property.

        This test verifies the behavior of the std_tensor property in the ImageContext class.
        It checks that an exception is raised if the std_tensor property is accessed before 
        it is set. It also verifies that the std_tensor property can be set and retrieved 
        correctly.

        Assertions:
        - An exception is raised when accessing std_tensor before it is set.
        - The std_tensor property can be set to a torch.Tensor value.
        - The std_tensor property returns the correct tensor value.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        with self.assertRaises(AttributeError) as context_exception:
            _ = context.std_tensor
        self.assertIn("Object property `std_tensor` has not yet been set", str(context_exception.exception))
        context.std_tensor = torch.tensor([0.2, 0.2, 0.2])
        self.assertTrue(torch.allclose(context.std_tensor, torch.tensor([0.2, 0.2, 0.2]), atol=1e-08))



    def test_invalid_std_tensor_property(self):
        """
        Test setting std_tensor property with an invalid type.

        This test verifies that an exception is raised when attempting to set the 
        std_tensor property with a value that is not of type torch.Tensor. The 
        exception message should indicate the required type for the std_tensor property.

        Assertions:
        - An exception is raised with a message indicating the invalid type for std_tensor.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        with self.assertRaises(Exception) as context_exception:
            context.std_tensor = "invalid_tensor"
        self.assertIn("Property `std_tensor` must be of type `<class 'torch.Tensor'>`", str(context_exception.exception))

    def test_check_all_pil_images_objects(self):
        """
        Test _check_all_pil_images_objects method with valid images.

        This test verifies that the _check_all_pil_images_objects method passes 
        without raising an exception when all images generated by the image loader 
        are valid PIL images.

        Assertions:
        - No exception is raised when _check_all_pil_images_objects is called.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        context._check_all_pil_images_objects()  # Should not raise exception

    def test_check_all_pil_images_objects_invalid(self):
        """
        Test _check_all_pil_images_objects method with invalid images.

        This test verifies that an exception is raised when the _check_all_pil_images_objects 
        method encounters an invalid image that is not of type PIL.Image or PIL.PngImageFile.
        The exception message should indicate the invalid image types.

        Assertions:
        - An exception is raised with a message indicating the invalid image types.
        """
        def invalid_image_generator():
            yield 'This is wrong'
        mock_loader = MagicMock(spec=LetterImageLoader)
        mock_loader.gen_all_images.side_effect = invalid_image_generator() 

        with self.assertRaises(Exception) as context_exception:
            ImageContext(image_loader=mock_loader, image_mode='rgb')
        self.assertIn(
            "Some or all images loaded by `LetterImageLoader` object are not of types:", 
            str(context_exception.exception)
        )

    def test_check_input_image_mode_valid(self):
        """
        Test _check_input_image_mode method with valid image modes.

        This test verifies that the _check_input_image_mode method passes 
        without raising an exception when all images generated by the image loader 
        have the expected image mode (in this case, 'rgb').

        Assertions:
        - No exception is raised when _check_input_image_mode is called with valid image modes.
        """
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        context._check_input_image_mode()  # Should not raise exception

    def test_check_input_image_mode_invalid(self):
        """
        Test _check_input_image_mode method with invalid image modes.

        This test verifies that an exception is raised when the _check_input_image_mode 
        method encounters an image with an unexpected image mode (in this case, 'L' 
        for grayscale). The exception message should indicate the expected and actual 
        image modes.

        Assertions:
        - An exception is raised with a message indicating the mismatch between the expected 
          and actual image modes.
        """
        def invalid_mode_image_generator():
            yield Image.new('L', (100, 100))  # Grayscale image
        self.mock_loader.gen_all_images.return_value = invalid_mode_image_generator()
        context = ImageContext(image_loader=self.mock_loader, image_mode='rgb')
        with self.assertRaises(Exception) as context_exception:
            context._check_input_image_mode()
            self.assertIn("Expected `image_mode` equal to `rgb`, but input image has mode 'L'", str(context_exception.exception))


class TestImageMixin(unittest.TestCase):

    def setUp(self):
        
        mock_loader = MagicMock(spec=LetterImageLoader)
        def image_generator(letter_test=None):
            for _ in range(5):
                yield Image.new('RGB', (100, 100))

        mock_loader.gen_all_images.side_effect = image_generator
        mock_loader.gen_letter_images.side_effect = image_generator
        self.mock_loader = mock_loader
        self.mock_icontext = ImageContext(image_loader=self.mock_loader, image_mode = 'rgb')
        self.image_generator_callable = image_generator


    def test_gen_img_batch_as_nparray(self):
        '''
        Test is verifying that the batch size is as expected when we request a batch of size 2.
        The test checks
          - Total number of images produces is 5, so the batches should of size [2,2,1]
          - Also checks that a `StopIteration` exception is raised after the final batch is retrieved.
        '''

        batch_size = 2
        batch_gen = ImageMixin._gen_img_batch_as_nparray(self.mock_icontext, batch_size)
        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], batch_size)

        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], batch_size)
        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], 1)

        with self.assertRaises(StopIteration) as execption_context:
            next(batch_gen)

    def test_gen_all_img_as_nparray(self):
        '''
        Test is verifying that the batch size is as expected when we request a batch of size 0.
        The test checks
          - Total number of images produced is 5, which should be all in one batch.
          - Also checks that a `StopIteration` exception is raised after the final batch is retrieved.
        '''

        batch_size = 0
        batch_gen = ImageMixin._gen_img_batch_as_nparray(self.mock_icontext, batch_size)
        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], 5)

        with self.assertRaises(StopIteration) as execption_context:
            next(batch_gen)

    def test_gen_img_letter_batch_as_nparray(self):
        '''
        Test is verifying that the batch size is as expected when we request a batch of size 2 for a specific letter.
        The test checks:
          - Total number of images produced is 5, so the batches should be of size [2, 2, 1]
          - Also checks that a `StopIteration` exception is raised after the final batch is retrieved.
        '''

        batch_size = 2
        letter = 'A'
        batch_gen = ImageMixin._gen_img_letter_batch_as_nparray(self.mock_icontext, batch_size, letter)
        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], batch_size)

        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], batch_size)
        
        batch = next(batch_gen)
        self.assertEqual(batch.shape[0], 1)

        with self.assertRaises(StopIteration):
            next(batch_gen)

    def test_gen_img_letter_batch_as_nparray_invalid_letter(self):
        '''
        Test is verifying that an exception is raised when an invalid letter is passed.
        '''

        batch_size = 2
        invalid_letter = '1'
        
        with self.assertRaises(Exception) as context:
            batch_gen = ImageMixin._gen_img_letter_batch_as_nparray(self.mock_icontext, batch_size, invalid_letter)
            next(batch_gen)
        
        self.assertIn('Unknown letter type passed to argument `letter` with value: 1', str(context.exception))


if __name__ == '__main__':
    unittest.main()
