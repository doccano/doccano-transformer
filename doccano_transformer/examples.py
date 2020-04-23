from typing import Callable, List, Optional


class Example:
    def is_valid(self, raise_exception: Optional[bool] = True) -> None:
        raise NotImplementedError


class NERExample:
    def __init__(self) -> None:
        ...

    def is_valid(self, raise_exception: Optional[bool] = True) -> None:
        ...

    def to_spacy(self, tokenizer: Callable[[str], List[str]]) -> dict:
        ...
