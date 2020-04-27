import csv
import json
from typing import Any, Callable, Iterable, Iterator, List, Optional, TextIO

from doccano_transformer.examples import Example, NERExample


class Dataset:
    def __init__(
        self,
        filepath: str,
        encoding: Optional[str] = 'utf-8',
        transformation_func: Optional[Callable[[TextIO], Iterable[Any]]] = None
    ) -> None:

        self.filepath = filepath
        self.encoding = encoding
        self.transformation_func = transformation_func or (lambda x: x)

    def __iter__(self) -> Iterator[Any]:
        with open(self.filepath, encoding=self.encoding) as f:
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
