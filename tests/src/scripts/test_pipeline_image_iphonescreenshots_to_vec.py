import unittest
from unittest.mock import MagicMock
import torch
from PIL import Image, ImageOps
import numpy as np
from typing import Generator
from src.scripts.pipeline_image_iphonescreenshots_to_vec import ImageContext, PytorchTensorImageNormalizationStrategy 



class TestImageContext(unittest.TestCase):

    def test_single_rgb_image(self):
        '''
        Testing to see if single rgb image has the right
        - shape
        - number of examples
        - correct image mode
        
        '''

        img = Image.new('RGB', (100, 100))
        context = ImageContext(images=img, image_mode='rgb')
        self.assertEqual(context.image_mode, 'rgb')
        self.assertEqual(context.num_of_examples, 1)
        for data in context.data:
            self.assertEqual(data.shape, (100,100,3))

    def test_single_grayscale_image(self):
        '''
        Testing to see if single grayscale image has the right
        - shape
        - number of examples
        - correct image mode
        '''
        
        img = Image.new('L', (100, 100))
        context = ImageContext(images=img, image_mode='grayscale')
        self.assertEqual(context.image_mode, 'grayscale')
        self.assertEqual(context.num_of_examples, 1)
        for data in context.data:
            self.assertEqual(data.shape, (100, 100))

    def test_generator_of_images(self):
        '''
        Testing to see if multiple rgb image generator produces objects with
        - right size
        - number of examples
        - correct image mode
        - output tensor values as expected based on provided inputs
        
        '''

        def image_generator() -> Generator[Image.Image, None, None]:
            for i in range(5):
                yield Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
        
        context = ImageContext(images=image_generator, image_mode='rgb')
        self.assertEqual(context.image_mode, 'rgb')
        self.assertEqual(context.num_of_examples, 5)
        for i, data in enumerate(context.data):
            self.assertEqual(data.shape, (100, 100, 3))
            self.assertTrue((np.array(data) == (i * 50)).all())

    def test_invalid_image_mode(self):
        '''
        Testing to see what happens when the wrong image_mode parameter value is provided. 
        Values currently set for 'rgb' and 'grayscale'
        '''
        img = Image.new('RGB', (100, 100))
        with self.assertRaises(Exception) as context:
            ImageContext(images=img, image_mode='invalid_mode')
        self.assertIn('Unknown `image_mode` parameter', str(context.exception))

    def test_non_pil_images_in_generator(self):
        '''
        Testing to see if generator produces the expected datatype
        '''
        def invalid_image_generator() -> Generator[int, None, None]:
            for i in range(5):
                yield i
        
        with self.assertRaises(Exception) as context:
            ImageContext(images=invalid_image_generator, image_mode='rgb')
        self.assertIn('Some elements of iterator passed to arg `images` are not of types', str(context.exception))




class TestPytorchTensorImageNormalizationStrategy(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.strategy = PytorchTensorImageNormalizationStrategy(batch_size=self.batch_size)

    def create_activate_image_generator(self, num_images, mode='RGB'):
        mode = mode
        def generator():
            for i in range(num_images):
                if mode == 'RGB':
                    color = (i * 10, i * 10, i * 10)
                elif mode == 'L':
                    color = (i*10)
                yield Image.new(mode, (10, 10), color=color)
        return generator()

    def test_call_with_invalid_context(self):
        with self.assertRaises(Exception) as context:
            self.strategy(None)
        self.assertTrue("Argument `context` must be of type" in str(context.exception))

    def test_call_with_valid_context(self):
        image_gen = self.create_activate_image_generator(10)
        
        def image_generator() -> Generator[Image.Image, None, None]:
            def gen():    
                for i in range(5):
                    yield Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
            return gen()

        context = ImageContext(images=image_generator, image_mode='rgb')
        self.strategy(context)


    def test_handle_rgb_and_grayscale_mean_calc(self):
        image_gen = self.create_activate_image_generator(10)
        expected_num_examples = 10
        mean_tensor = self.strategy._handle_rgb_and_grayscale_mean_calc(
            image_gen,
            expected_num_examples
        )
        self.assertEqual(mean_tensor.shape, torch.Size([3]))
        
        image_gen = self.create_activate_image_generator(10, mode="L")
        expected_num_examples = 10
        mean_tensor = self.strategy._handle_rgb_and_grayscale_mean_calc(
            image_gen,
            expected_num_examples
        )
        self.assertEqual(mean_tensor.shape, torch.Size([]))


    def test_get_batch(self):
        image_gen = self.create_activate_image_generator(4)
        batches = list(self.strategy._get_batch(image_gen, self.batch_size))

        self.assertEqual(len(batches), 2)  # 4 images in total, batch size is 2
        self.assertEqual(batches[0].shape, (10, 10, 3))
        self.assertEqual(batches[1].shape, (10, 10, 3))

    def test__get_mean_tensor_values_rgb(self):

        def create_image_generator_rgb(
            num_images: int, width: int, height: int
        ) -> Generator[Image.Image, None, None]:
            for i in range(num_images):
                yield Image.fromarray(np.full((height, width, 3), fill_value=(i * 10, i*10, i*10), dtype=np.uint8))

        image_gen = create_image_generator_rgb(3,2,2) 
        mock_context = MagicMock(spec=ImageContext)
        mock_context.data = image_gen
        mock_context.num_of_examples = 3
        mock_context.image_mode = 'rgb'
        mean_result_from_func = self.strategy._get_mean_tensor(mock_context)
        
        image_gen = create_image_generator_rgb(3,2,2) 
        mock_context.reset_data_generator = MagicMock(return_value=image_gen)
        mock_context.data = mock_context.reset_data_generator()
        examples = torch.tensor(
            np.array(
                [np.array(i) for i in create_image_generator_rgb(3,2,2)]
            )
        )
        sum_by_example = torch.sum(examples, axis=0)
        sum_by_channel = torch.sum(sum_by_example, axis=(0,1))
        W, H = examples.shape[1], examples.shape[2]
        mean_result_from_manual = sum_by_channel / (examples.shape[0]*W*H)
   
        self.assertTrue(torch.allclose(mean_result_from_manual, mean_result_from_func, rtol=1e-04, atol=1e-05))
   

    def test__get_std_tensor_values_rgb(self):

        def image_generator(r=20) -> Generator[Image.Image, None, None]:
            for i in range(r):
                yield Image.new('RGB', (100, 100), color=(i * 50, i * 50, i * 50))
        
        context = ImageContext(images=image_generator, image_mode='rgb')

        mean_result_from_func = self.strategy._get_mean_tensor(context)
        context.data = context.reset_data_generator()

        std_result_from_func = self.strategy._get_variance_tensor(
            context, 
            mean_result_from_func
        )

        examples = torch.tensor(
            np.array(
                [np.array(i) for i in image_generator(20)]
            ),
            dtype=torch.float32
        )
        num_of_pixels = examples.shape[0] * examples.shape[1] * examples.shape[2] 
        std_result_from_manual = torch.sqrt(
            (
                (examples - mean_result_from_func)**2).sum(axis=[0,1,2])/num_of_pixels
        )

        self.assertTrue(torch.allclose(std_result_from_func, std_result_from_manual, rtol=1e-04, atol=1e-05))





if __name__ == '__main__':
    unittest.main()





