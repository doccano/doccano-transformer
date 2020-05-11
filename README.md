# doccano-transformer

[![Build Status](https://github.com/doccano/doccano-transformer/workflows/CI/badge.svg)](https://github.com/doccano/doccano-transformer/actions)

Doccano Transformer helps you to transform an exported dataset into the format of your favorite machine learning library.

## Support formats

Doccano Transformer supports the following formats:

* CoNLL 2003(NER)
* spaCy(NER)

## Install

To install `doccano-transformer`, simply use `pip`:

```bash
$ pip install doccano-transformer
```

## Examples

### CoNLL 2003(NER)

```python
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

d = read_jsonl(filepath='example.jsonl', dataset=NERDataset, encoding='utf-8')
d.to_conll2003(tokenizer=str.split)
```

### SpaCy(NER)

```python
d = read_jsonl(filepath='example.jsonl', dataset=NERDataset, encoding='utf-8')
d.spacy(tokenizer=str.split)
```
