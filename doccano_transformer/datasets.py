import csv
import json
from typing import Any, Callable, Iterable, Iterator, List, Optional, TextIO

from doccano_transformer.examples import Example, NERExample
from doccano_transformer.utils import from_spacy


class Dataset:
    def __init__(
        self,
        filepath: str,
        encoding: Optional[str] = 'utf-8',
        transformation_func: Optional[Callable[[TextIO], Iterable[Any]]] = None,
        user_id: Optional[int] = None,
    ) -> None:

        self.filepath = filepath
        self.encoding = encoding
        self.transformation_func = transformation_func or (lambda x: x)
        self.user_id = user_id

    def __iter__(self) -> Iterator[Any]:
        with open(self.filepath, encoding=self.encoding) as f:
            if self.user_id:
                yield from self.transformation_func(f,self.user_id)
            else:
                yield from self.transformation_func(f)

    @classmethod
    def from_jsonl(
            cls, filepath: str, encoding: Optional[str] = 'utf-8'
    ) -> 'Dataset':
        return cls(filepath, encoding, lambda f: map(json.loads, f))

    @classmethod
    def from_csv(
        cls, filepath: str, encoding: Optional[str] = 'utf-8'
    ) -> 'Dataset':
        return cls(filepath, encoding, csv.DictReader)

    @classmethod
    def from_spacy(
        cls, filepath: str, encoding: Optional[str] = 'utf-8',user_id: Optional[int]=1
    ) -> 'Dataset':
        return cls(filepath, encoding, from_spacy, user_id)

class TaskDataset(Dataset):
    example_class: Example = None

    def __iter__(self) -> Iterator[Example]:
        for raw in super(TaskDataset, self).__iter__():
            example = self.example_class(raw)
            example.is_valid(raise_exception=True)
            yield example


class NERDataset(TaskDataset):
    example_class = NERExample

    def to_conll2003(
        self, tokenizer: Callable[[str], List[str]]
    ) -> Iterator[str]:
        for example in self:
            yield from example.to_conll2003(tokenizer)

    def to_spacy(
        self, tokenizer: Callable[[str], List[str]]
    ) -> Iterator[dict]:
        for example in self:
            yield from example.to_spacy(tokenizer)

    def to_jsonl(
        self,
    ) -> Iterator[dict]:
        for example in self:
            yield from example.to_jsonl()