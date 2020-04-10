from collections import defaultdict
from enum import Enum
from typing import Callable, List, Optional

from doccano_transformer import utils


class NotSupportedInputFormatError(Exception):
    pass


class NotSupportedOutputFormatError(Exception):
    pass


class InputFormat:

    @staticmethod
    def valid(x: dict) -> bool:
        raise NotImplementedError


class NER(InputFormat):

    FIELDS = ['id', 'text', 'meta', 'annotations', 'annotation_approver']

    @classmethod
    def valid(cls, x: dict) -> bool:
        # TODO: This method should be stricter.
        for field in cls.FIELDS:
            if field not in x:
                return False
        return True

    def __init__(self, x: dict, tokenizer: Callable[[str], List[str]]) -> None:
        if not self.valid(x):
            raise NotSupportedInputFormatError

        self.id = x['id']
        self.text = x['text']
        self.tokens = tokenizer(x['text'])
        self.offsets = utils.get_offsets(self.text, self.tokens)
        self.meta = x['meta']
        labels = defaultdict(list)
        for annotation in x['annotations']:
            labels[annotation['user']].append([
                annotation['start_offset'],
                annotation['end_offset'],
                annotation['label']
            ])
        self.labels = labels
        self.annotation_approver = x['annotation_approver']
        self.default_user = max(labels.keys(), key=lambda user: len(labels[user]))

    def to_conll2003(self, user: Optional[int] = None) -> str:
        label = self.labels[user or self.default_user]
        if not label:
            return None
        tags = utils.create_bio_tags(self.tokens, self.offsets, label)
        lines = []
        for token, tag in zip(self.tokens, tags):
            lines.append(f'{token} _ _ {tag}\n')
        lines.append('\n')
        return ''.join(lines)


class NERTextLabel(NER):

    FIELDS = ['id', 'text', 'meta', 'labels', 'annotation_approver']

    def __init__(self, x: dict, tokenizer: Callable[[str], List[str]]) -> None:
        if not self.valid(x):
            raise NotSupportedInputFormatError

        self.id = x['id']
        self.text = x['text']
        self.tokens = tokenizer(x['text'])
        self.offsets = utils.get_offsets(self.text, self.tokens)
        self.meta = x['meta']
        labels = defaultdict(list)
        for label in x['labels']:
            # TODO: This format doesn't have a user field currently.
            # So this method uses the user 0 for all label.
            labels[-1].append(label)
        self.labels = labels
        self.annotation_approver = x['annotation_approver']
        self.default_user = -1


class OutputFormat(Enum):
    CSV = 0
    CoNLL2003 = 1
    JSON = 2
    JSONL = 3
    SpaCy = 4
