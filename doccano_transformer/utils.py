from typing import Optional

from doccano_transformer.datasets import Dataset


def read_jsonl(
        filepath: str,
        dataset: Dataset,
        encoding: Optional[str] = 'utf-8'
) -> Dataset:
    return dataset.from_jsonl(filepath, encoding)


def read_csv(
        filepath: str,
        dataset: Dataset,
        encoding: Optional[str] = 'utf-8'
) -> Dataset:
    return dataset.from_csv(filepath, encoding)
