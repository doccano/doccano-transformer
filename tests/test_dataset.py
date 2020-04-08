import os
from unittest import TestCase

import pytest

from doccano_transformer import Dataset
from doccano_transformer.tasks import NER


class TestDataset(TestCase):

    @pytest.fixture(autouse=True)
    def initdir(self, shared_datadir, tmpdir):
        self.shared_datadir = shared_datadir
        self.tmpdir = tmpdir

    def test_to_conll2003(self):
        src_path = self.shared_datadir / 'labeling_text_label.jsonl'
        tgt_path = self.tmpdir / 'labeling_text_label.conll'
        d = Dataset(filepath=src_path, task=NER)
        d.to_conll2003(tgt_path)

        self.assertTrue(os.path.isfile(tgt_path))
