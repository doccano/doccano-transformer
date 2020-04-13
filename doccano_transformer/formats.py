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
        self.sentences = utils.split_sentences(x['text'])
        self.sentence_offsets = utils.get_offsets(x['text'], self.sentences)
        self.sentence_offsets.append(len(x['text']))
        self.tokens = [tokenizer(sentence) for sentence in self.sentences]
        self.token_offsets = [
            utils.get_offsets(sentence, tokens, offset)
            for sentence, tokens, offset in zip(
                self.sentences, self.tokens, self.sentence_offsets
            )
        ]
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
        self.default_user = max(
            labels.keys(), key=lambda user: len(labels[user])
        )

    def to_conll2003(self, user: Optional[int] = None) -> str:
        labels = self.labels[user or self.default_user]
        if not labels:
            return None
        label_split = [[] for _ in range(len(self.sentences))]
        for label in labels:
            for i, (start, end) in enumerate(
                    zip(self.sentence_offsets, self.sentence_offsets[1:])):
                if start <= label[0] <= label[1] <= end:
                    label_split[i].append(label)

        lines = ['-DOCSTART- -X- -X- O\n\n']
        for tokens, offsets, label in zip(
                self.tokens, self.token_offsets, label_split):
            tags = utils.create_bio_tags(tokens, offsets, label)
            for token, tag in zip(tokens, tags):
                lines.append(f'{token} _ _ {tag}\n')
            lines.append('\n')
        return ''.join(lines)


class NERTextLabel(NER):

    FIELDS = ['id', 'text', 'meta', 'labels', 'annotation_approver']

    def __init__(self, x: dict, tokenizer: Callable[[str], List[str]]) -> None:
        if not self.valid(x):
            raise NotSupportedInputFormatError

        self.id = x['id']
        self.sentences = utils.split_sentences(x['text'])
        self.sentence_offsets = utils.get_offsets(x['text'], self.sentences)
        self.sentence_offsets.append(len(x['text']))
        self.tokens = [tokenizer(sentence) for sentence in self.sentences]
        self.token_offsets = [
            utils.get_offsets(sentence, tokens, offset)
            for sentence, tokens, offset in zip(
                self.sentences, self.tokens, self.sentence_offsets
            )
        ]
        self.meta = x['meta']
        labels = defaultdict(list)
        for label in x['labels']:
            # TODO: This format doesn't have a user field currently.
            # So this method uses the user -1 for all label.
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
