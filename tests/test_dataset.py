from unittest import TestCase

import pytest

from doccano_transformer import Dataset
from doccano_transformer.tasks import NER


class TestDataset(TestCase):

    @pytest.fixture(autouse=True)
    def initdir(self, shared_datadir, tmpdir):
        self.shared_datadir = shared_datadir
        self.tmpdir = tmpdir

    def test_from_labeling_text_label_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling_text_label.jsonl'
        tgt_path = self.tmpdir / 'labeling_text_label.conll2003'
        expected = list(open(
            self.shared_datadir / 'labeling_text_label.conll2003'
        ))
        d = Dataset(filepath=src_path, task=NER)
        d.to_conll2003(tgt_path)

        self.assertListEqual(list(open(tgt_path)), expected)

    def test_from_labeling_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling.jsonl'
        tgt_path = self.tmpdir / 'labeling.conll2003'
        expected = list(open(
            self.shared_datadir / 'labeling.conll2003'
        ))
        d = Dataset(filepath=src_path, task=NER)
        d.to_conll2003(tgt_path)

        self.assertListEqual(list(open(tgt_path)), expected)
