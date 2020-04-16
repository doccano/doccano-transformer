import os
from typing import Callable, List, Optional

from . import utils
from .formats import NotSupportedOutputFormatError, OutputFormat
from .tasks import Task


class Dataset:
    def __init__(self,
                 filepath: str,
                 task: Task,
                 tokenizer: Optional[Callable[[str], List[str]]] = str.split
                 ) -> None:
        """
        Args:
            filepath (str): The path to the exported Doccano file.
            task (Task): The annotation task of the exported file.
            tokenizer (Optional[Callable[[str], List[str]]]): The tokenizer.
        """

        if not os.path.isfile(filepath):
            raise FileNotFoundError

        self.task = task
        self.data = task.load(filepath, tokenizer)

    def to_conll2003(self, savepath: str) -> None:
        """Convert the exported Doccano file to CoNLL2003 format.
        Args:
            savepath (str): The path to save the file converted to
                CoNLL2003 format.
        """
        if OutputFormat.CoNLL2003 not in self.task.allowed_output_formats:
            raise NotSupportedOutputFormatError
        else:
            use_suffix = len(self.data.users) > 1
            for user in self.data.users:
                converted_data = self.data.to_conll2003(user)
                if use_suffix:
                    utils.save_to_text(
                        converted_data, savepath + f'.user{user}'
                    )
                else:
                    utils.save_to_text(converted_data, savepath)

    def to_spacy(self, savepath: str) -> None:
        """Convert the exported Doccano file to CoNLL2003 format.
        Args:
            savepath (str): The path to save the file converted to
                CoNLL2003 format.
        """
        if OutputFormat.SpaCy not in self.task.allowed_output_formats:
            raise NotSupportedOutputFormatError
        else:
            use_suffix = len(self.data.users) > 1
            for user in self.data.users:
                converted_data = self.data.to_spacy(user)
                if use_suffix:
                    utils.save_to_json(
                        converted_data, savepath + f'.user{user}'
                    )
                else:
                    utils.save_to_json(converted_data, savepath)
