import csv
import json
from typing import List, Tuple


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


def create_bio_tags(
        tokens: List[str],
        offsets: List[int],
        labels: List[Tuple[int, int, str]]) -> List[str]:
    """Create BIT tags from Doccano's label data.
    Args:
        tokens (List[str]): The list of the token.
        offsets (List[str]): The list of the character offset.
        labels (List[Tuple[int, int, str]]): The list of labels. Each item in
            the list holds three values which are the start offset, the end
            offset, and the label name.
    Returns:
        (List[str]): The list of the BIO tag.
    """
    labels = sorted(labels)
    n = len(labels)
    i = 0
    prefix = 'B-'
    tags = []
    for token, token_start in zip(tokens, offsets):
        token_end = token_start + len(token)
        if i >= n or token_end < labels[i][0]:
            tags.append('O')
        elif token_start > labels[i][1]:
            tags.append('O')
        else:
            tags.append(prefix + str(labels[i][2]))
            if labels[i][1] > token_end:
                prefix = 'I-'
            elif i < n:
                i += 1
                prefix = 'B-'
    return tags
