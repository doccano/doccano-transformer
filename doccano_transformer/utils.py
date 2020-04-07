import csv
import json
from typing import List


def load_from_jsonl(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return list(map(json.loads, f))


def load_from_csv(filepath: str) -> List[dict]:
    with open(filepath) as f:
        return list(csv.DictReader(f))


def save_to_text(lines: List[str], filepath: str) -> None:
    with open(filepath, 'w') as f:
        f.writelines(lines)
