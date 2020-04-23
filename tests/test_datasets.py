from unittest import TestCase

import pytest

from doccano_transformer.datasets import NERDataset


class TestNERDataset(TestCase):

    @pytest.fixture(autouse=True)
    def initdir(self, shared_datadir, tmpdir):
        self.shared_datadir = shared_datadir / 'labeling'
        self.tmpdir = tmpdir

    def test_from_labeling_text_label_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling_text_label.jsonl'
        tgt_path = self.tmpdir / 'labeling_text_label.conll2003'
        expected = list(open(
            self.shared_datadir / 'labeling_text_label.conll2003'
        ))
        d = NERDataset(filepath=src_path)
        d.to_conll2003(tgt_path, str.split)

        self.assertListEqual(list(open(tgt_path)), expected)

    def test_from_labeling_jsonl_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling.jsonl'
        filename = 'labeling.conll2003'
        tgt_path = self.tmpdir / filename
        expected = sorted(self.shared_datadir.glob(f'{filename}.*'))

        d = NERDataset(filepath=src_path)
        d.to_conll2003(tgt_path, str.split)
        for y in expected:
            self.assertEqual(
                (self.tmpdir / y.name).read_text(None),
                y.read_text()
            )

    def test_from_labeling_jsonl_to_spacy(self):
        src_path = self.shared_datadir / 'labeling_multi_users.jsonl'
        filename = 'labeling_multi_users.spacy'
        tgt_path = self.tmpdir / filename
        expected = sorted(self.shared_datadir.glob(f'{filename}.*'))

        d = NERDataset(filepath=src_path)
        d.to_spacy(tgt_path, str.split)
        for y in expected:
            self.assertEqual(
                (self.tmpdir / y.name).read_text(None),
                y.read_text()
            )
