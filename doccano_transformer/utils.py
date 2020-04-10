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


def get_offsets(text: str, tokens: List[str]) -> List[int]:
    """Calculate char offsets of each tokens.

    Args:
        text (str): The string before tokenized.
        tokens (List[str]): The list of the string. Each string corresponds
            token.
    Returns:
        (List[str]): The list of the offset.
    """
    offsets = []
    m, n = map(len, (text, tokens))
    i = j = 0
    for token in tokens:
        for j, char in enumerate(token):
            while char != text[i]:
                i += 1
            if j == 0:
                offsets.append(i)
    return offsets
