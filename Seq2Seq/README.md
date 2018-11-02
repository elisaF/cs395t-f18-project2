# Sequence to sequence models for paraphrase generation

## Transformer

### Overview

* Implementing Vaswani et al. (2017).
* Full annotation, following full-phrase variable naming lint.

### Usage

* Place a folder `Data/` as the same folder as the `.py` scripts. `Data/` should contain four `.txt` files: `train_source.txt`, `train_target.txt`, `valid_source.txt`, `valid_target.txt`. Each line of the `.txt` is a sentence (words separated with whitespace), with no start/end symbols (these will be added while data reading).
* Place a folder `SavedModels/` as the same folder as the `.py` scripts. Leave it empty on the first round of training. Later it contains a `.ckpt` model state dict, a `.p` indexer (bidirectional word<->index mapping) file.
* Place a folder `Glove/` as the same folder as the `.py` scripts, which should contain a download glove `.txt` file ([download link](https://nlp.stanford.edu/projects/glove/)).
* To run: `sh run_trainer.sh` with desired configs.

### Variant 1. + Linear Context

* The context encoding treats every context sentence the same way as it does with the source sentence. The multiple encodings are finally concatenated and projected to the size of that of the source sentence on its own.
* Speed ~0.65x factors the base Transformer.

### Variant 2. + LSTM Context

* The context encoding treats context sentences as a sequence which are concatenated (separated by START/END symbols). The sequence is then processed with a stacked BiLSTM. We take the final state (h) as the context encoding.
* Speed ~0.05x factors the base Transformer.
