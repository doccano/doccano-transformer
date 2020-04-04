import csv
import json
import os
from typing import List


class NotDoccanoFormatError(Exception):
    pass


DOCCANO_KEYS = ('text', 'labels', 'meta')


def check_exists(filepath: str) -> None:
    if not os.path.isfile(filepath):
        raise FileNotFoundError


def is_valid_jsonl(filepath: str) -> bool:
    try:
        with open(filepath) as f:
            for x in map(json.loads, f):
                for key in DOCCANO_KEYS:
                    if key not in x:
                        raise KeyError
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def is_valid_csv(filepath: str) -> bool:
    try:
        with open(filepath) as f:
            for x in csv.DictReader(f):
                for key in DOCCANO_KEYS:
                    if key not in x:
                        raise KeyError
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def from_jsonl(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return list(map(json.loads, f))


def from_csv(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return list(csv.DictReader(f))
