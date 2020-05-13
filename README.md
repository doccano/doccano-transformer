# doccano-transformer

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9fe17d104b644a53a3fe189433d3c797)](https://app.codacy.com/gh/doccano/doccano-transformer?utm_source=github.com&utm_medium=referral&utm_content=doccano/doccano-transformer&utm_campaign=Badge_Grade_Dashboard)
[![Build Status](https://github.com/doccano/doccano-transformer/workflows/CI/badge.svg)](https://github.com/doccano/doccano-transformer/actions)

Doccano Transformer helps you to transform an exported dataset into the format of your favorite machine learning library.

## Supported formats

Doccano Transformer supports the following formats:

* CoNLL 2003
* spaCy

## Install

To install `doccano-transformer`, simply use `pip`:

```bash
pip install doccano-transformer
```

## Examples

### Named Entity Recognition

The following formats are supported:

- CoNLL 2003
- spaCy

```python
from doccano_transformer.datasets import NERDataset
from doccano_transformer.utils import read_jsonl

dataset = read_jsonl(filepath='example.jsonl', dataset=NERDataset, encoding='utf-8')
dataset.to_conll2003(tokenizer=str.split)
dataset.to_spacy(tokenizer=str.split)
```

## Contribution

We encourage you to contribute to doccano transformer! Please check out the [Contributing to doccano transformer guide](https://github.com/doccano/doccano-transformer/blob/master/CONTRIBUTING.md) for guidelines about how to proceed. 

## License

[MIT](https://github.com/doccano/doccano-transformer/blob/master/LICENSE)
