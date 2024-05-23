import io
import asyncio
import aiofiles
import numpy as np
from abc import ABC, abstractmethod
from typing import (
    Optional, Tuple,
    Generator, AsyncGenerator,
    List,
)
from PIL import Image
from src.fs.training_data_locator import AlphabetImagesLocator, AbsTrainDataLocator 
import string

class AbstractLoaderCreator(ABC):

    @abstractmethod
    def factory_method(self, main_dir: Optional[str] = None):
        raise NotImplemented()

class LetterImageLoaderCreator(AbstractLoaderCreator):

    @classmethod
    def factory_method(cls, main_dir: Optional[str] = None):
        main_dir = main_dir if main_dir else 'images'
        data_locator = AlphabetImagesLocator(main_dir)
        return LetterImageLoader(data_locator = data_locator)


class LetterImageLoader:

    def __init__(self, data_locator: AlphabetImagesLocator):
        if not issubclass(data_locator.__class__, AbsTrainDataLocator):
            raise Exception(f"data_locator arg must be of subclass of {AbsTrainDataLocator}")

        self.data_locator = data_locator

    def _get_letter_image_paths(
            self, 
            letter: str,
            *extra_filters
    ) -> List[str]:
        files = list(
            filter(
                lambda path: '/' + letter + '/' in path , 
                self.data_locator.get_paths()
            )
        )

        if not files:
            def print_handler():
                for i in self.data_locator.get_paths():
                    if '/U/' in i: print(i)

            raise Exception(
                f"No files found for letter {letter} at path: {print_handler()}"
            )

        if extra_filters:
            for _filter in extra_filters:
                files = list(
                    filter(
                        lambda path: _filter in path, 
                        files                    
                    )
                )

        return files


    def get_letter_images(
        self, 
        letter: str, 
        *args
    ) -> List[Image.Image]:
        files = self._get_letter_image_paths(letter, *args) 
        images = [Image.open(file) for file in files]
        
        return images


    def gen_letter_images(
        self, 
        letter: str,
        *args
    ) -> Generator[Image.Image, None, None]:
         
        files = self._get_letter_image_paths(letter, *args) 
        for file in files:
            yield Image.open(file)

    def gen_all_images(self) -> Generator[Image.Image, None, None]:
        files = self.data_locator.get_paths()
        for file in files:
            yield Image.open(file)

    def gen_all_images_by_letter(self) -> Generator[Tuple[Image.Image, str], None, None]:
        for letter in (string.ascii_uppercase + string.ascii_lowercase):
            files = self.data_locator.get_letter_files_path(letter)

            for file in files:
                img = Image.open(file)
                yield (img, letter)


    async def async_gen_letter_images(
        self, 
        letter: str, 
        *args
    ) -> AsyncGenerator[Image.Image, None]:
        files = self._get_letter_image_paths(letter, *args)

        for file in files:
            async with aiofiles.open(file, 'rb') as fp:
                data = await fp.read()
                yield Image.open(io.BytesIO(data)) 


