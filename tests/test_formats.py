from unittest import TestCase

from doccano_transformer import formats


class TestNER(TestCase):

    def test_valid_returns_true(self):
        x = {'id': 0, 'text': 'Nadim Ladki', 'meta': {},
             'annotation_approver': {},
             'annotations': [
            {'label': 2, 'start_offset': 0, 'end_offset': 5, 'user': 1},
            {'label': 4, 'start_offset': 6, 'end_offset': 11, 'user': 2}
        ]}
        self.assertTrue(formats.NER.valid(x))

    def test_valid_returns_false(self):
        x = {'id': 0, 'text': 'Nadim Ladki', 'meta': {},
             'annotation_approver': {}}
        self.assertFalse(formats.NER.valid(x))

    def test_to_conll2003(self):
        x = {'id': 0, 'text': 'Nadim Ladki', 'meta': {},
             'annotation_approver': {},
             'annotations': [
            {'label': 2, 'start_offset': 0, 'end_offset': 5, 'user': 1},
            {'label': 4, 'start_offset': 6, 'end_offset': 11, 'user': 2}
        ]}
        result = formats.NER(x, str.split).to_conll2003(user=1)
        expected = 'Nadim _ _ B-2\nLadki _ _ O\n\n'
        self.assertEqual(result, expected)

        result = formats.NER(x, str.split).to_conll2003(user=2)
        expected = 'Nadim _ _ O\nLadki _ _ B-4\n\n'
        self.assertEqual(result, expected)


class TestNERTextLabel(TestCase):
    def test_to_conll2003(self):
        x = {"id": 2576, 'text': 'Nadim Ladki', 'meta': {},
             'annotation_approver': {}, 'labels': [[0, 11, 'PER']]}
        result = formats.NERTextLabel(x, str.split).to_conll2003(user=-1)
        expected = 'Nadim _ _ B-PER\nLadki _ _ I-PER\n\n'
        self.assertEqual(result, expected)
