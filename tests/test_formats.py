from unittest import TestCase

from doccano_transformer import formats


class TestNERFormat(TestCase):

    def test_to_conll2003(self):
        text = 'Nadim Ladki'
        annotations = [
            {'label': 2, 'start_offset': 0, 'end_offset': 5, 'user': 1},
            {'label': 4, 'start_offset': 6, 'end_offset': 11, 'user': 1}
        ]

        x = formats.NERFormat(
            id=0, text=text, annotations=annotations,
            meta={}, annotation_approver={}
        )
        expected = 'Nadim\t2\nLadki\t4\n\n'

        self.assertEqual(x.to_conll2003(), expected)
