# cs395t-f18-project2

## Contextualized Paraphrase Generation

### Overview

This work explores paraphrasing with context, i.e. given a source sentences and a few consecutive sentences that come immediately before the source, predict a paraphrase for the source that is coherent with the context. 

### Data

Taking the ROCstories dataset (Mostafazadeh/17) as the basis, we crowdsource gold paraphrases for the 5th sentence and take the 1-4th sentences as the context.

### Pretrained Models

* Word embeddings: GloVe (Pennington/14).
* Event embeddings: recoding sentences as events with Weber/18.

### Paraphasing Models

* Base Transformer (Vaswani/17): the model performs well in regular paraphrasing tasks in both automatic and human evaluation (Wang/19).
* Linear-context Transformer: encoding the context sentences in the same way we do the source sentence, and use a linear layer to project the extended encoding to the size of the original source sentence embeddings. 
* LSTM-context Transformer: encoding the context as a sequence of sentences using a stacked BiLSTM, then use the final hidden state as the context encoding. The context encoding is concatenated to the embeddings of each token in the source sentence for information augmentation. The augmented embeddings are then projected with a linear layer to the size of the original source sentence embeddings.
