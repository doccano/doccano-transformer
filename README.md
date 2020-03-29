# doccano-transformer

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
from doccano_transformer import Transformer
from doccano_transformer.tasks import NER

t = Transformer(filepath='example.jsonl', task=NER)
t.to_spacy()
```
