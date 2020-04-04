from typing import Callable, List, Optional

from . import utils
from .tasks import Task


class Dataset:
    def __init__(self,
                 filepath: str,
                 task: Task,
                 tokenizer: Optional[Callable[[str], List[str]]] = str.split):

        utils.check_exists(filepath)

        if utils.is_valid_jsonl(filepath):
            self.data = utils.from_jsonl(filepath)
        elif utils.is_valid_csv(filepath):
            self.data = utils.from_csv(filepath)
        else:
            raise utils.NotDoccanoFormatError

        self.task = task
        self.tokenizer = tokenizer

    def to_conll2003(self, savepath: str):
        ...
