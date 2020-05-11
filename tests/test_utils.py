from unittest import TestCase

from doccano_transformer import utils


class TestUtils(TestCase):

    def test_get_offsets(self):
        text = ' This is Doccano Transformer . '
        tokens = text.split()
        result = utils.get_offsets(text, tokens)
        expected = [1, 6, 9, 17, 29]
        self.assertListEqual(result, expected)

    def test_create_bio_tags(self):
        tokens = ' This is Doccano Transformer . '.split()
        offsets = [1, 6, 9, 17, 29]
        labels = [[9, 28, 'SOFTWARE']]
        result = utils.create_bio_tags(tokens, offsets, labels)
        expected = ['O', 'O', 'B-SOFTWARE', 'I-SOFTWARE', 'O']
        self.assertListEqual(result, expected)

    def test_convert_tokens_and_offsets_to_spacy_tokens(self):
        tokens = 'This is Doccano Transformer .'.split()
        offsets = [0, 5, 8, 16, 28]
        spacy_tokens = utils.convert_tokens_and_offsets_to_spacy_tokens(
            tokens, offsets
        )
        for i, (spacy_token, token, offset) in enumerate(
            zip(spacy_tokens, tokens, offsets)
        ):
            self.assertEqual(str(spacy_token), token)
            self.assertEqual(len(spacy_token), len(token))
            self.assertEqual(spacy_token.i, i)
            self.assertEqual(spacy_token.idx, offset)
