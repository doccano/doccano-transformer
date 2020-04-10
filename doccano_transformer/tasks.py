from typing import Callable, List, Optional

from . import formats, utils


class Task:

    @staticmethod
    def load(filepath: str) -> 'Task':
        raise NotImplementedError


class NER(Task):

    allowed_output_formats = {
        formats.OutputFormat.CoNLL2003,
        formats.OutputFormat.SpaCy
    }

    def __init__(
        self, data: List[dict], tokenizer: Callable[[str], List[str]]
    ) -> None:
        if all(formats.NER.valid(x) for x in data):
            self.data = [formats.NER(x, tokenizer) for x in data]
        elif all(formats.NERTextLabel.valid(x) for x in data):
            self.data = [formats.NERTextLabel(x, tokenizer) for x in data]
        else:
            raise formats.NotSupportedInputFormatError

    @classmethod
    def load(cls,
             filepath: str,
             tokenizer: Callable[[str], List[str]]
             ) -> 'NER':
        raw_data: List[dict] = utils.load_from_jsonl(filepath)
        return cls(raw_data, tokenizer)

    def to_conll2003(self, user: Optional[int] = None) -> List[str]:
        result = ['-DOCSTART- -X- -X- O\n\n']
        for x in self.data:
            line = x.to_conll2003(user)
            if not line:
                continue
            result.append(line)
        return result
