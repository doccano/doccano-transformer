# doccano-transformer

[![Build Status](https://github.com/doccano/doccano-transformer/workflows/CI/badge.svg)](https://github.com/doccano/doccano-transformer/actions)

Doccano Transformer helps you to transform an exported dataset into the format of your favorite machine learning library.

## Support formats

Doccano Transformer supports the following formats:

* CSV
* CoNLL 2003
* JSON
* JSON Lines
* SpaCy

## Install

To install `doccano-transformer`, simply use `pip`:

```bash
$ pip install doccano-transformer
```

## Examples

```python
from doccano_transformer import Dataset
from doccano_transformer.tasks import NER

d = Dataset(filepath='example.jsonl', task=NER, tokenizer=str.split)
d.to_spacy()
```
