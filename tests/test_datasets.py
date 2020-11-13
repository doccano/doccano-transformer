import json
from collections import defaultdict
from pathlib import Path
from unittest import TestCase

from doccano_transformer.datasets import NERDataset


class TestNERDataset(TestCase):
    @classmethod
    def setUp(self):
        current_path = Path(__file__).parent
        self.shared_datadir = current_path / 'data' / 'labeling'

    def test_from_labeling_text_label_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling_text_label.jsonl'
        filename = 'labeling_text_label.conll2003'
        users = defaultdict(list)
        d = NERDataset.from_jsonl(filepath=src_path)
        for x in d.to_conll2003(str.split):
            users[x['user']].append(x['data'])

        for user, data in users.items():
            with open(self.shared_datadir / (filename + f'.user{user}')) as f:
                expected = f.read()
            self.assertEqual(''.join(data), expected)

    def test_from_labeling_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling.jsonl'
        filename = 'labeling.conll2003'
        users = defaultdict(list)
        d = NERDataset.from_jsonl(filepath=src_path)
        for x in d.to_conll2003(str.split):
            users[x['user']].append(x['data'])

        for user, data in users.items():
            with open(self.shared_datadir / (filename + f'.user{user}')) as f:
                expected = f.read()
            self.assertEqual(''.join(data), expected)

    def test_from_labeling_text_label_jsonl_to_spacy(self):
        src_path = self.shared_datadir / 'labeling_text_label.jsonl'
        filename = 'labeling_text_label.spacy'
        users = defaultdict(list)
        d = NERDataset.from_jsonl(filepath=src_path)
        for x in d.to_spacy(str.split):
            users[x['user']].append(x['data'])

        for user, data in users.items():
            with open(self.shared_datadir / (filename + f'.user{user}')) as f:
                expected = json.load(f)
            # print(data)
            self.assertEqual(data, expected)

    def test_from_labeling_jsonl_to_spacy(self):
        src_path = self.shared_datadir / 'labeling.jsonl'
        filename = 'labeling.spacy'
        users = defaultdict(list)
        d = NERDataset.from_jsonl(filepath=src_path)
        for x in d.to_spacy(str.split):
            users[x['user']].append(x['data'])

        for user, data in users.items():
            with open(self.shared_datadir / (filename + f'.user{user}')) as f:
                expected = json.load(f)
            self.assertEqual(data, expected)
