import unittest
import os
from unittest.mock import patch, MagicMock
from src.fs.training_data_locator import AlphabetImagesLocator

class TestAlphabetImagesLocator(unittest.TestCase):

    @patch('os.listdir')
    def setUp(self, mock_listdir):
        self.main_dir = 'test_dir'
        mock_listdir.return_value = ['A', 'B', 'C', 'a', 'b', 'c']
        self.locator = AlphabetImagesLocator(self.main_dir)
        self.mock_locator = MagicMock(spec=AlphabetImagesLocator)

    @patch('os.listdir')
    def test_get_paths(self, mock_listdir):
        '''
        Verify that the file paths returned as as expected based on dummy values provided. 
        '''
        # Mock the listdir to return dummy files
        mock_listdir.side_effect = [
            ['file1.png', 'file2.png'],  # for lower case 'a'
            [],  # for lower case 'b'
            [],  # for lower case 'c'
            [],  # for upper case 'A'
            [],  # for upper case 'B'
            [],  # for upper case 'C'
            # ... add more empty lists to cover all alphabet directories
        ] + [[]] * (len(self.locator._all_dirs) - 6)
        
        paths = self.locator.get_paths()
        expected_paths = [
            os.path.join(self.locator._lc_dirs[0], 'file1.png'),
            os.path.join(self.locator._lc_dirs[0], 'file2.png')
        ]

        
        self.assertEqual(paths, expected_paths)


    def test_get_root_path(self):
        '''
        Verify that the project root path in the object matches the one at the class level. 
        '''
        self.assertEqual(self.locator.get_root_path(), AlphabetImagesLocator.project_data_dir)


    def test_get_letter_subdir_path(self):
        '''
        Verify the path to a letter-level subdirectory is as expected 
        '''
        expected_path = os.path.join(self.locator._main_lc_dir, 'a')
        self.assertEqual(self.locator.get_letter_subdir_path('a'), expected_path)

        expected_path = os.path.join(self.locator._main_uc_dir, 'A')
        self.assertEqual(self.locator.get_letter_subdir_path('A'), expected_path)

    @patch('os.listdir')
    def test_get_letter_files_path(self, mock_listdir):

        mock_listdir.side_effect = [
            [
                os.path.join(self.locator._uc_dirs[0], 'file1.png'),
                os.path.join(self.locator._uc_dirs[0], 'file2.png')
            ]
        ]

        expected_paths = [
            os.path.join(self.locator._uc_dirs[0], 'file1.png'),
            os.path.join(self.locator._uc_dirs[0], 'file2.png')
        ]
        self.assertEqual(self.locator.get_letter_files_path('A'), expected_paths)

    
    def test_get_letter_files_path_raise_dir_not_found_exception(self,):
        '''
        Validating that execption is raised when no letter sub dir is found.
        Test works because although the locator object has director names, none of
        them currently end with numbers. All of the end with a case-sensitive letter. 
        '''
        with self.assertRaises(Exception) as context:
            self.locator.get_letter_files_path('324234')
        self.assertIn("No sub directory found for letter", str(context.exception))

    @patch('os.listdir')
    def test_get_letter_files_path_raise_no_images_found_exception(self, mock_listdir):
        '''
        Validating that execption is raised when no images are found for letter 'A'
        Returning an empty list when we search the subdirectory for letter 'A' for image files.
        '''

        mock_listdir.side_effect = [[]]
        letter = 'A'

        with self.assertRaises(Exception) as context:
            self.locator.get_letter_files_path(letter)
        self.assertIn(f'No files found for letter {letter} at directory ', str(context.exception))

 


if __name__ == '__main__':
    unittest.main()
