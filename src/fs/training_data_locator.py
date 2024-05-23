import os
from abc import ABC, abstractmethod
from typing import List

class AbsTrainDataLocator(ABC):
    
    project_data_dir = '/Users/hassanmahmood/repos/mynotes-coreml/data'
    
    @abstractmethod
    def get_paths(self) -> List[str]:
        pass
   
    @classmethod
    @abstractmethod
    def get_root_path(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_training_data_options(cls) -> List[str]:
        pass

class AlphabetImagesLocator(AbsTrainDataLocator):

    _lower_letters_dir = 'letters/lower'
    _upper_letters_dir = 'letters/upper'
    _upper_alphabet = [chr(i) for i in range(65, 91)]  # Uppercase A-Z
    _lower_alphabet = [chr(i) for i in range(97, 123)]  #


    
    def __init__(self, main_dir):
        self._main_dir = os.path.join(self.project_data_dir, main_dir)
        
        self._main_lc_dir = os.path.join(self._main_dir, self._lower_letters_dir)
        self._main_uc_dir = os.path.join(self._main_dir, self._upper_letters_dir)
        
        self._lc_dirs = [os.path.join(self._main_lc_dir, letter) for letter in self._lower_alphabet]
        self._uc_dirs = [os.path.join(self._main_uc_dir, letter) for letter in self._upper_alphabet]
        self._all_dirs = self._lc_dirs + self._uc_dirs

    @classmethod
    def get_root_path(cls):
        return cls.project_data_dir

    @classmethod
    def get_training_data_options(cls) -> List[str]:
        return os.listdir(cls.project_data_dir)

    def get_paths(self) -> List[str]:
        all_paths = []
        for subdir in self._all_dirs:
            filenames = os.listdir(subdir)
            paths = [os.path.join(subdir, filename) for filename in filenames ]
            all_paths.extend(paths)

        return all_paths

    def get_letter_subdir_path(
        self,
        letter: str
    ) -> str:
        search_term = f'/{letter}'
        for dir in self._all_dirs:
            if dir.endswith(search_term):
                return dir

        raise Exception(f"Directory path for `letter` {letter} not found for search term `{search_term}`. ")
        
    def get_letter_files_path(
        self,
        letter: str
    ) -> List[str] :
        search_term = f'/{letter}'
        letter_dir = ''
        for dir in self._all_dirs:
            if dir.endswith(search_term):
                letter_dir = dir
                break

        if not letter_dir:
            raise Exception(f'No sub directory found for letter {letter} in private variable `_alldirs`')
       
        files = [os.path.join(letter_dir, file) for file in os.listdir(letter_dir)]

        if not files:
            raise Exception(f'No files found for letter {letter} at directory {letter_dir}. ')

        return files

