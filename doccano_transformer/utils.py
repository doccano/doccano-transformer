from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from doccano_transformer.datasets import Dataset


def read_jsonl(
        filepath: str,
        dataset: 'Dataset',
        encoding: Optional[str] = 'utf-8'
) -> 'Dataset':
    return dataset.from_jsonl(filepath, encoding)


def read_csv(
        filepath: str,
        dataset: 'Dataset',
        encoding: Optional[str] = 'utf-8'
) -> 'Dataset':
    return dataset.from_csv(filepath, encoding)


def split_sentences(text: str) -> List[str]:
    return text.split('\n')


def get_offsets(
        text: str,
        tokens: List[str],
        start: Optional[int] = 0) -> List[int]:
    """Calculate char offsets of each tokens.

    Args:
        text (str): The string before tokenized.
        tokens (List[str]): The list of the string. Each string corresponds
            token.
        start (Optional[int]): The start position.
    Returns:
        (List[str]): The list of the offset.
    """
    offsets = []
    i = 0
    for token in tokens:
        for j, char in enumerate(token):
            while char != text[i]:
                i += 1
            if j == 0:
                offsets.append(i + start)
    return offsets


def create_bio_tags(
        tokens: List[str],
        offsets: List[int],
        labels: List[Tuple[int, int, str]]) -> List[str]:
    """Create BI tags from Doccano's label data.

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


class Token:
    def __init__(self, token: str, offset: int, i: int) -> None:
        self.token = token
        self.idx = offset
        self.i = i

    def __len__(self):
        return len(self.token)

    def __str__(self):
        return self.token


def convert_tokens_and_offsets_to_spacy_tokens(
    tokens: List[str], offsets: List[int]
) -> List[Token]:
    """Convert tokens and offsets to the list of SpaCy compatible object.

    Asrgs:
        tokens (List[str]): The list of tokens.
        offsets (List[int]): The list of offsets.
    Returns:
        (List[Token]): The list of the SpaCy compatible object.
    Examples:
        >>> from doccano_transformer import utils
        >>> tokens = ['This', 'is', 'Doccano', 'Transformer', '.']
        >>> offsets = [0, 5, 8, 16, 28]
        >>> utils.convert_tokens_and_offsets_to_spacy_tokens(tokens, offsets)
    """
    if len(tokens) != len(offsets):
        raise ValueError('tokens size should equal to offsets size')
    spacy_tokens = []
    for i, (token, offset) in enumerate(zip(tokens, offsets)):
        spacy_tokens.append(Token(token, offset, i))
    return spacy_tokens
