import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from typing import (
    get_type_hints,
    Generator, Tuple
)
from src.fs.training_data_locator import AlphabetImagesLocator
from src.fs.loader import LetterImageLoader
from PIL import Image
import inspect


class TestLetterImageLoader(unittest.TestCase):

    def setUp(self):
        self.mock_locator = MagicMock(spec=AlphabetImagesLocator)
        self.mock_locator.get_paths.return_value = [
            '/path/to/A/file1.png',
            '/path/to/A/file2.png',
            '/path/to/B/file1.png',
            '/path/to/U/file1.png',
            '/path/to/U/file2.png',
        ]
        self.loader = LetterImageLoader(self.mock_locator)


    def test_initialization_with_valid_locator(self):
        '''
        Validating that the correct type is mocked and allowed during instantiation.
        '''
        try:
            loader = LetterImageLoader(self.mock_locator)
        except Exception as e:
            self.fail(f"Initialization with valid locator raised an exception: {e}")


    def test_initialization_with_invalid_locator(self):
        '''
        Validating that an execption is raised when a data type other than the expected
        `LetterImageLoader` type is passed to initializer.
        '''
        with self.assertRaises(Exception) as context:
            loader = LetterImageLoader(object())
        self.assertIn("data_locator arg must be of subclass of", str(context.exception))


    def test_get_letter_image_paths(self):
        '''
        Validating that when we requests a target letter, we only get the directory paths
        for that letter and the form is as expected. 
        '''
        letter = 'A'
        expected_paths = [
            '/path/to/A/file1.png',
            '/path/to/A/file2.png'
        ]
        paths = self.loader._get_letter_image_paths(letter)
        self.assertEqual(paths, expected_paths)


    def test_get_letter_image_paths_with_extra_filters(self):
        '''
        Verify that when passed extra filters to refine the search, the function can do that. 
        It is returning back now a single file path for the subdir A, which should contain 2 files. 
        The test is selecting `file1.png`.
        '''
        letter = 'A'
        extra_filter = 'file1.png'
        expected_paths = [
            '/path/to/A/file1.png', 
        ]
        paths = self.loader._get_letter_image_paths(letter, extra_filter)
        self.assertEqual(paths, expected_paths)


    def test_get_letter_image_paths_no_files_found(self):
        '''
        Test that an exception is raised when no files are found for a given letter. This is crucial as the 
        function cannot return an empty list or a non-type. It MUST produce an execption if the results of 
        the search at negative. 

        We have mocked our data locator object to not return a 'Z' letter path. 
        '''
        letter = 'Z'
        with self.assertRaises(Exception) as context:
            self.loader._get_letter_image_paths(letter)
        self.assertIn("No files found for letter", str(context.exception))


    @patch('src.utils.image_utils.Image.open')  # Replace 'your_module' with the actual module name
    def test_get_letter_images(self, mock_image_open):
        ''' 
        Test that we are getting 
        
        1. The correct number of images from the sub directory for letter 'A'
        2. We are getting the right data type from the method `get_letter_images`, which is a 
            `List[Image.Image]`
        '''

        letter = 'A'
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image
       
        # verifying return type is generator
        images = self.loader.get_letter_images(letter)
        self.assertIsInstance(images, list)

        expected_paths = [
            '/path/to/A/file1.png',
            '/path/to/A/file2.png'
        ]

        # check we have the right number of images found for 'A'
        self.assertEqual(len(images), len(expected_paths))

        # check that the return types are of type 'Image.Image'
        for image in images:
            self.assertIs(image, mock_image)


    @patch('src.utils.image_utils.Image.open')  
    def test_gen_all_images(self, mock_image_open):
        ''' 
        Test that we are getting a generator to generate all the letter images, one by one
        
        1. The correct number of images are returned, which should be all the images
        2. We are getting the right data type from the method `get_all_images`, which is a 
            generator
        3. We are also verifiy the Generator is returning the right datatype. 

        '''
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        image_generator = self.loader.gen_all_images()
        
        # Check if the returned object is a generator
        self.assertTrue(inspect.isgenerator(image_generator))

        # Check if the generator type is Generator[Image.Image, None, None]
        gen_type = get_type_hints(self.loader.gen_all_images)['return']
        self.assertEqual(gen_type, Generator[Image.Image, None, None])

        images = list(image_generator)
        expected_paths = [
            '/path/to/A/file1.png',
            '/path/to/A/file2.png',
            '/path/to/B/file1.png',
            '/path/to/U/file1.png',
            '/path/to/U/file2.png'
        ]

        self.assertEqual(len(images), len(expected_paths))
        for image in images:
            self.assertIs(image, mock_image)


    @patch('src.utils.image_utils.Image.open')  
    def test_gen_all_images_by_letter(self, mock_image_open):
        '''
        We are testing to verify that we recieve a generator for generating images for all letter,
        such that the returned type from the generator is of type `Tuple[Image.Image, str]` where 
        the string represents the letter i.e. (image, 'A') . 
        The test is looking to verify:
            
            1. The returned type of the function call is a generator
            2. The generator yield `Tuple[Image.Imgae, str]`
            3. The structure of the tuple is as we expect it to be. 
        '''

        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image

        # Mock get_letter_files_path to return file paths
        def get_letter_files_path_mock(letter):
            if letter == 'A':
                return ['/path/to/A/file1.png', '/path/to/A/file2.png']
            elif letter == 'B':
                return ['/path/to/B/file1.png']
            else:
                return []
        
        self.mock_locator.get_letter_files_path.side_effect = get_letter_files_path_mock

        image_generator = self.loader.gen_all_images_by_letter()

        # Check if the returned object is a generator
        self.assertTrue(inspect.isgenerator(image_generator))

        # Check if the generator type is Generator[Tuple[Image.Image, str], None, None]
        gen_type = get_type_hints(self.loader.gen_all_images_by_letter)['return']
        self.assertEqual(gen_type, Generator[Tuple[Image.Image, str], None, None])

        images = list(image_generator)
        expected_images = [
            (mock_image, 'A'),
            (mock_image, 'A'),
            (mock_image, 'B')
        ]
        
        # Validate that the structure of the returned tuple is as expected
        self.assertEqual(len(images), len(expected_images))
        for image, expected in zip(images, expected_images):
            img, letter = image
            exp_img, exp_letter = expected
            self.assertIs(img, exp_img)
            self.assertEqual(letter, exp_letter)

    @patch('src.utils.image_utils.aiofiles.open', new_callable=AsyncMock)
    @patch('src.utils.image_utils.Image.open')
    async def test_async_gen_letter_images(self, mock_image_open, mock_aiofiles_open):
        '''
        Testing the async letter-specific image generator

        Testing to see if:
            1. Returned type from async method is a `AsyncGenerator` type
            2. The returned images for letter 'A' are of expected size
            3. The type of value returned by generator is correct e.g. Image.Image
        '''
        mock_image = MagicMock(spec=Image.Image)
        mock_image_open.return_value = mock_image
        
        mock_fp = AsyncMock()
        mock_fp.read.return_value = b'test_image_data'

        async def mock_open(file, mode='r', *args, **kwargs):
            return mock_fp

        mock_aiofiles_open.side_effect = mock_open

        # Mock _get_letter_image_paths to return file paths
        self.loader._get_letter_image_paths = MagicMock(return_value=['/path/to/A/file1.png', '/path/to/A/file2.png'])

        image_generator = self.loader.async_gen_letter_images('A')

        # Check if the returned object is an async generator
        self.assertTrue(inspect.isasyncgen(image_generator))

        images = [image async for image in image_generator]

        # Check if the images object is a list and has the expected images
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 2)
        for img in images:
            self.assertIs(img, mock_image)


if __name__ == '__main__':
    unittest.main()
