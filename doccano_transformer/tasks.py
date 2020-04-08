from typing import List, NamedTuple

from . import formats, utils


class Task:

    @staticmethod
    def load(filepath: str) -> List[NamedTuple]:
        raise NotImplementedError


class NER(Task):

    allowed_output_formats = {
        formats.OutputFormat.CoNLL2003,
        formats.OutputFormat.SpaCy
    }

    def __init__(self, data: List[dict]) -> None:
        try:
            self.data = [formats.NERFormat(**x) for x in data]
        except TypeError:
            self.data = [formats.NERTextLabelFormat(**x) for x in data]
        except TypeError:
            raise formats.NotSupportedInputFormatError

    @classmethod
    def load(cls, filepath: str) -> List[NamedTuple]:
        raw_data = utils.load_from_jsonl(filepath)
        return cls(raw_data)

    def to_conll2003(self) -> List[str]:
        return [x.to_conll2003() for x in self.data]
