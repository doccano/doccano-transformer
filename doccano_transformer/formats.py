import os
from collections import namedtuple
from enum import Enum


class NotSupportedInputFormatError(Exception):
    pass


class NotSupportedOutputFormatError(Exception):
    pass


class NERFormat(namedtuple(
    'NERFormat',
    ['id', 'text', 'meta', 'annotations', 'annotation_approver']
)):

    def to_conll2003(self):
        annotations = self.annotations
        text = self.text
        lines = []
        for annotation in annotations:
            start_offset = annotation['start_offset']
            end_offset = annotation['end_offset']
            token = text[start_offset: end_offset]
            tag = annotation['label']
            lines.append(f'{token}\t{tag}{os.linesep}')
        return ''.join(lines) + os.linesep


class NERTextLabelFormat(namedtuple(
    'NERTextLabelFormat',
    ['id', 'text', 'meta', 'labels', 'annotation_approver']
)):

    def to_conll2003(self):
        labels = self.labels
        text = self.text
        lines = []
        for start, end, tag in labels:
            token = text[start: end]
            lines.append(f'{token}\t{tag}{os.linesep}')
        return ''.join(lines) + os.linesep


class OutputFormat(Enum):
    CSV = 0
    CoNLL2003 = 1
    JSON = 2
    JSONL = 3
    SpaCy = 4
