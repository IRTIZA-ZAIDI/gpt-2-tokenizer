# gpt-2-tokenizer-main

A small, readable implementation of **byte-level BPE tokenization** (GPT-style), plus a **GPT-4 compatible tokenizer wrapper** that matches `tiktoken`’s `cl100k_base` encoding.

This repo is useful if you want to:

* understand how GPT-style BPE works end-to-end
* train a tiny tokenizer on custom text
* experiment with regex-based splitting and special tokens
* verify GPT-4 tokenization behavior without pulling in a large codebase

---

## What’s included

### Tokenizers (`minbpe/`)

* **`BasicTokenizer`**: minimal byte-level BPE tokenizer (no regex splitting, no special tokens).
* **`RegexTokenizer`**: byte-level BPE tokenizer with:

  * GPT-style regex text splitting (GPT-4 pattern by default)
  * optional **special token** handling (similar to `tiktoken`’s `allowed_special`)
  * save/load support (`.model` + `.vocab`)
* **`GPT4Tokenizer`**: a pretrained tokenizer wrapper that reconstructs GPT-4 merges from `tiktoken`’s `cl100k_base` and matches its encoding outputs (including special tokens).

### Learning materials (`learning/`)

Contains notebook(s) and artifacts used for experimentation and understanding:

* `learning/tokenizer.ipynb`
* reference files like `vocab.bpe`, `encoder.json`, and a small toy corpus.

### Tests (`tests/`)

* `tests/test_tokenizer.py` validates:

  * encode/decode identity
  * GPT-4 equality vs `tiktoken`
  * special tokens behavior
  * save/load correctness
* `tests/pakistan_wiki.txt` is used as a larger text sample.

---

## Project structure

```
minbpe/
  __init__.py
  base.py          # common utilities + Tokenizer base (save/load)
  basic.py         # BasicTokenizer
  regex.py         # RegexTokenizer (GPT2/GPT4 split patterns + special tokens)
  gpt4.py          # GPT4Tokenizer wrapper around RegexTokenizer + tiktoken
learning/
  tokenizer.ipynb
  vocab.bpe
  encoder.json
  encoder.py
  openai_public.py
  tok400.model
  tok400.vocab
  toy.txt
tests/
  test_tokenizer.py
  pakistan_wiki.txt
```

---

## Requirements

This repo uses:

* Python 3.9+ (recommended: 3.10+)
* `regex` (the third-party regex engine, not the stdlib `re`)
* `tiktoken` (for GPT-4 / `cl100k_base` parity in `GPT4Tokenizer`)
* `pytest` (for tests)

---

## Installation

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows PowerShell

pip install --upgrade pip
pip install regex tiktoken pytest
```

If you want to import `minbpe` from anywhere, you can also do an editable install:

```bash
pip install -e .
```

Note: there is no `setup.py/pyproject.toml` included in this repo snapshot. If `pip install -e .` fails, just run your scripts/notebooks from the repo root so Python can import `minbpe/`.

---

## Quickstart

### 1) BasicTokenizer (no regex splitting, no special tokens)

```python
from minbpe import BasicTokenizer

text = "hello hello world"
tok = BasicTokenizer()

tok.train(text, vocab_size=256 + 50)  # 50 merges
ids = tok.encode("hello world")
print(ids)
print(tok.decode(ids))
```

### 2) RegexTokenizer (GPT-style splitting + optional special tokens)

```python
from minbpe import RegexTokenizer

text = "hello world!!!? (안녕하세요!) lol123"
tok = RegexTokenizer()               # defaults to GPT-4 split pattern

tok.train(text, vocab_size=256 + 100)
ids = tok.encode(text)
print(tok.decode(ids))
```

#### Special tokens

```python
from minbpe import RegexTokenizer

tok = RegexTokenizer()
tok.train("some training text", vocab_size=256 + 50)

special_tokens = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
}
tok.register_special_tokens(special_tokens)

s = "<|endoftext|>Hello"
ids = tok.encode(s, allowed_special="all")
print(ids)
print(tok.decode(ids))
```

`allowed_special` behavior:

* `"all"`: allow any registered special token
* `"none"`: treat special tokens as ordinary text
* `"none_raise"`: raise/assert if special tokens appear in input (default behavior)
* `set([...])`: allow only a subset of special tokens

---

## Saving and loading a trained tokenizer (RegexTokenizer)

After training:

```python
from minbpe import RegexTokenizer

tok = RegexTokenizer()
tok.train("a bit of text to train on", vocab_size=256 + 64)

tok.save("my_tokenizer")        # writes my_tokenizer.model and my_tokenizer.vocab
```

Later:

```python
from minbpe import RegexTokenizer

tok = RegexTokenizer()
tok.load("my_tokenizer.model")

ids = tok.encode("test string")
print(tok.decode(ids))
```

---

## GPT-4 tokenizer parity (via `tiktoken`)

`GPT4Tokenizer` is a pretrained wrapper that matches `tiktoken`’s `cl100k_base`:

```python
from minbpe import GPT4Tokenizer
import tiktoken

text = "hello world"
tok = GPT4Tokenizer()

ids_local = tok.encode(text)
ids_official = tiktoken.get_encoding("cl100k_base").encode(text)

assert ids_local == ids_official
print(ids_local)
```

Special tokens:

```python
text = "<|endoftext|>Hello"
tok = GPT4Tokenizer()
ids = tok.encode(text, allowed_special="all")
print(ids)
print(tok.decode(ids))
```

Note:

* `GPT4Tokenizer` is not intended to be trained.
* It cannot be saved/loaded with the base save/load format (by design in `minbpe/gpt4.py`).

---

## Running tests

From repo root:

```bash
pytest -q
```

The tests cover:

* encode/decode identity for multiple tokenizers
* equality with `tiktoken` for GPT-4 (`cl100k_base`)
* special token handling correctness
* save/load roundtrip for `RegexTokenizer`
* a reference BPE merge example from Wikipedia

---

## Notebook

Open:

* `learning/tokenizer.ipynb`

This is the best place to explore tokenization behavior interactively and see how merges/vocab evolve during training.

---

## Notes / limitations

* This is a minimal educational implementation focused on clarity.
* Training is implemented for `BasicTokenizer` and `RegexTokenizer`.
* GPT-4 parity is achieved by reconstructing merges from `tiktoken`’s internal ranks and applying byte shuffling quirks that exist in the official encoding.

