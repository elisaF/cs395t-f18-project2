# trainer_linear_context.py

import argparse
import copy
import dill
from loader_linear_context import *
import math
import numpy as np
import os
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from transformer_linear_context import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearContextBatch:
    """Object for holding a batch of data with mask during training."""
    
    def __init__(self, source, contexts, target=None, pad=0):
        """
        Args:
            source: shape = <batch-size, seq-length>.
            contexts: a list where each element has the same shape as the source.
            target: shape = <batch-size, seq-length>.
            pad: padding index.
        """
        self.source = source
        # Make mask: <batch-size, seq-length> -> <batch-size, 1, seq-length>
        self.source_mask = (source != pad).unsqueeze(-2)
        self.contexts, self.context_masks = [], []
        for i, context in enumerate(contexts):
            self.contexts.append(context)
            self.context_masks.append((context != pad).unsqueeze(-2))
        if target is not None:
            self.target = target[:, :-1]  # cut the last column, 0 to -1
            self.target_y = target[:, 1:] # cut the first column, 1 to last
            self.target_mask = \
                self.make_standard_mask(self.target, pad)
            # Var number_tokens: int, batch-size*(seq-length -1)
            self.number_tokens = (self.target_y != pad).data.sum()
    
    @staticmethod
    def make_standard_mask(target, pad):
        "Create a mask to hide padding and future words."
        # Make mask: <batch-size, seq-length-1> -> <batch-size, 1, seq-length-1>
        target_mask = (target != pad).unsqueeze(-2) 
        target_mask = target_mask & Variable(
            subsequent_mask(target.size(-1)).type_as(target_mask.data))
        # `target_mask` shape = <batch-size, seq-length, seq-length>
        return target_mask


def get_linear_context_batch(vocab_size, batch_size, number_batches, data_iterator):
    """Get a Batch() object for training.
    
    Args:
        vocab_size: vocab size.
        batch_size: batch size.
        number_batches: number of batches.
        data_iterator: a DataIterator object.
    Returns:
        A Batch object.
    """
    for i in range(number_batches):
        batch = data_iterator.random_batch(batch_size)
        source, target = batch[0], batch[-1]
        contexts = list(batch[1:-1])
        source, target = source.to(DEVICE), target.to(DEVICE)
        contexts = [context.to(DEVICE) for context in contexts]
        yield LinearContextBatch(source, contexts, target, pad=0)
        

def make_linear_context_model(source_vocab_size, target_vocab_size, 
                              number_blocks, embed_size, upsample_size, 
                              number_heads, dropout_rate, glove_init):
    """Helper: Construct a model from hyperparameters.
    
    Args:
        source_vocab_size: vocab size of the source data.
        target_vocab_size: vocab size of the target data.
        number_blocks: number of blocks.
        embed_size: embedding size.
        upsample_size: `blow-up` size in ffnn for upsampling.
        number_heads: number of attention heads.
        dropout_rate: dropout rate.
        glove_init: numpy.ndarray, initializing embeddings.
    Returns:
        An EncoderDecoder object.
    """
    deep_copy = copy.deepcopy
    attention = MultiHeadedAttention(number_heads, embed_size)
    ffnn = PositionwiseFeedForward(embed_size, upsample_size, dropout_rate)
    position = PositionalEncoding(embed_size, dropout_rate)
    model = LinearContextEncoderDecoder(
        Encoder(EncoderBlock(embed_size, 
                             deep_copy(attention), 
                             deep_copy(ffnn), 
                             dropout_rate), 
                number_blocks),
        Decoder(DecoderBlock(embed_size, 
                             deep_copy(attention), 
                             deep_copy(attention), 
                             deep_copy(ffnn), 
                             dropout_rate), 
                number_blocks),
        nn.Sequential(Embeddings(embed_size, source_vocab_size, glove_init), 
                      deep_copy(position)),
        nn.Sequential(Embeddings(embed_size, target_vocab_size, glove_init), 
                      deep_copy(position)),
        Generator(embed_size, target_vocab_size),
        embed_size=embed_size,
        context_size=4).to(DEVICE)
    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)
    return model


def run_epoch(batches, model, indexer, data_iterator, 
              loss_compute, print_every, data_group=""):
    """Standard Training and Logging Function
    
    Args:
        batches: Batch object (has many batches in side).
        model: EncoderDecoder object.
        indexer: Indexer object (for printing out sample predictions).
        data_iterator: DataIterator object (same purpose as above).
        loss_compute: SimpleLossCompute object.
        prnt_every: batch. Frequency of report printout.
        data_group: `"train"` or `"valid"`.
    Returns:
        Float average loss.
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    number_tokens = 0
    for i, batch in enumerate(batches):
        # Forward on
        #   source: <batch-size,seq-length>
        #   target: <batch-size, seq-length>
        #   source_mask: <batch-size, 1, seq-length>
        #   tgt_mask: <batch-size, seq-length, seq-length>
        # outputs: <batch-size, seq-length, embed-size>
        outputs = model.forward(batch.source, batch.contexts, batch.target, 
                                batch.source_mask, batch.context_masks, batch.target_mask)
        # Compute loss
        #   IN:
        #     outputs: <batch-size, seq-length=original-seq-length-1, embed-size>
        #     batch.target_y: outputs[:,1:]
        #   OUT:
        #     float loss
        loss = loss_compute(outputs, batch.target_y, batch.number_tokens)
        total_loss += loss
        total_tokens += batch.number_tokens
        number_tokens += batch.number_tokens
        if i != 0 and i % print_every == 0:
            random_predict(model, indexer, data_iterator)
            elapsed = time.time() - start         
            print("%s Step: %d Loss: %f | Tokens per Sec: %f (total time = %.2f)\n" %
                  (data_group, i, 
                   loss / batch.number_tokens.float(), 
                   number_tokens / elapsed, 
                   time.time()-start))
            start = time.time()
            number_tokens = 0
    return total_loss / total_tokens.float()


def linear_context_greedy_decode(model, source, contexts, source_mask, context_masks,
                                 max_length, start_symbol):
    """Decoding for the best prediction.
    
    Args:
        model: trained EncoderDecoder object.
        source: torch.LongTensor, <batch-size=1, seq-length>.
        contexts: a list of torch.LongTensor objects, each is of the same shape as `source`.
        source_mask: same type as source, <batch-size=1, 1, seq-length>.
        context_masks: a list of torch.LongTensor, each has the same shape as `source_mask`.
        max_length: maximal decoding length.
        start_symbol: <s>.
    Returns:
        torch.LongTensor prediction, <batch-size=1, seq-length>.
    """
    encoded_source = model.encode(source, source_mask)
    encoded_contexts = []
    for context, context_mask in zip(contexts, context_masks):
        encoded_contexts.append(model.encode(context, context_mask))
    memory = model.linear(torch.cat([encoded_source] + encoded_contexts, dim=-1))
    predictions = torch.ones(1, 1).fill_(start_symbol).type_as(source.data)
    for i in range(max_length-1):
        outputs = model.decode(memory, source_mask, 
                               Variable(predictions), 
                               Variable(subsequent_mask(predictions.size(1))
                                        .type_as(source.data)))
        prediction = model.generator(outputs[:, -1])
        _, next_word = torch.max(prediction, dim=1)
        next_word = next_word.item() # index for the next word.
        predictions = torch.cat([predictions, 
                                 torch.ones(1, 1).type_as(source.data).fill_(next_word)], 
                                dim=1)
        if next_word==3: # hard coded end symbol
            break
    return predictions


def tensor_to_words(indexer, tensor):
    """Convert a torch(.cuda).LongTensor() to a list of words.
    
    Args:
        indexer: Indexer object.
        tensor: torch.LongTensor, <batch-size=1, seq-length>
    Returns:
        A list of word strings.
    """
    if DEVICE.type != "cpu":
        tensor = tensor.cpu()
    return indexer.to_words(tensor.numpy()[0])


def random_predict(model, indexer, data_iterator, size=5):
    """Make predictions (and print out) for a `size` of random sentences.
    
    Args:
        model: EncoderDecoder object. Trained.
        indexer: Indexer object.
        data_iterator: DataIterator object.
        size: number of sentences tested.
    """
    print("Random predictions\n")
    for i in range(size):
        batch = data_iterator.random_batch(1)
        source, target = batch[0], batch[-1]
        contexts = list(batch[1:-1])
        source, target = source.to(DEVICE), target.to(DEVICE)
        contexts = [context.to(DEVICE) for context in contexts]    
        # Make mask: <batch-size=1, 1, seq-length>
        source_mask = Variable(torch.ones(1, 1, source.size(1))).to(DEVICE)
        context_masks = [Variable(torch.ones(1, 1, context.size(1))).to(DEVICE)
                         for context in contexts]
        prediction = linear_context_greedy_decode(model, source, contexts, 
                                                  source_mask, context_masks, 
                                                  max_length=target.size(1), 
                                                  start_symbol=2)
        source, prediction, target = [tensor_to_words(indexer, tensor) 
                                      for tensor in [source.data, prediction, target.data]]
        source = " ".join([token for token in source 
                           if token not in ["PAD","<s>","</s>"]])
        prediction = " ".join([token for token in prediction 
                               if token not in ["PAD","<s>","</s>"]])
        target = " ".join([token for token in target 
                           if token not in ["PAD","<s>","</s>"]])
        print("[Example %d]" % (i+1))
        print("SOURCE     >> " + source + "\n" + \
              "PREDICTION >> " + prediction + "\n" + \
              "TARGET     >> " + target + "\n")


def train(data_dir, model_dir, indexer_dir, glove_path,
          batch_size, number_batches, number_epochs, 
          embed_size, upsample_size, number_heads, number_blocks, factor, warmup, 
          learning_rate, dropout_rate,
          print_every=5, saved_model=None, model_name="transformer",
          validate=True):
    """Train the Transformer model.
    
    Args:
        data_dir: specified data directory with the following files with
                  the exact names:
                  `train_source.txt`, 
                  `train_context1.txt`, ..., `train_context4.txt`, 
                  `train_target.txt`,
                  `valid_source.txt`, 
                  `valid_context1.txt`, ..., `valid_context4.txt`, 
                  `valid_target.txt`.
                  Each of the files has each line as a sentence.
        model_dir: directory under which trained model is saved.
        glove_path: path to the glove .txt file downloaded from
                    `https://nlp.stanford.edu/projects/glove/`.
        batch_size: batch size.
        number_batches: number of random batches run each epoch.
        number_epochs: number of epochs.
        embed_size: embedding size.
        upsample_size: `blow-up` linear upsampling step (see Vaswani/17).
        number_heads: number of multi-attention heads.
        number_blocks: number of transformer encoder & decoder blocks.
        factor: rate of learning rate change (for Adam optimizer).
        warmup: learning rate increases before `warmup` steps then decreases.
        learning_rate: learning rate.
        dropout_rate: dropout rate.
        prnt_every: batch. Frequency of report printout within epoch.
        saved_model: path to the `.ckpt` model dict file.
        model_name: model name. String.
        validate: boolean. True to run validation.
    """
    # Load data
    print("Loading data\n")
    indexer, \
    train_source, \
    train_context1, train_context2, train_context3, train_context4, \
    train_target, \
    valid_source, \
    valid_context1, valid_context2, valid_context3, valid_context4, \
    valid_target = load_corpus(data_dir)
    print("Data loaded!\n")
    glove_init = load_glove(glove_path, indexer, embed_size)
    dill.dump(indexer, open(indexer_dir+"indexers.p", "wb"))
    print("Indexers saved!")
    print("\nDone!\n")
    # Prepare data
    vocab_size = indexer.size
    train_iterator = DataIterator(train_source, 
                                  train_context1, train_context2, train_context3, train_context4,
                                  train_target,
                                  len(train_source))
    valid_iterator = DataIterator(valid_source, valid_target,
                                  valid_context1, valid_context2, valid_context3, valid_context4,
                                  len(valid_source))
    model = make_linear_context_model(source_vocab_size=vocab_size, 
                                      target_vocab_size=vocab_size, 
                                      number_blocks=number_blocks, embed_size=embed_size, 
                                      upsample_size=upsample_size, number_heads=number_heads, 
                                      dropout_rate=dropout_rate, glove_init=glove_init)
    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
        print("Loaded saved model:", saved_model)
    criterion = LabelSmoothing(size=vocab_size, padding_index=0, smoothing=0.0)
    optimizer = NoamOpt(model.source_embed[0].embed_size, factor, warmup, 
                        torch.optim.Adam(model.parameters(), 
                                         lr=learning_rate, 
                                         betas=(0.9,0.98), eps=1e-9))
    print("Start training ...\n")
    best_valid_loss = np.inf
    number_batches_processed = 0
    train_losses = []
    for epoch in range(number_epochs):
        print( "Epoch %d\n" % (epoch+1))
        model.train() # training mode, gradient flows.
        train_loss = run_epoch(get_linear_context_batch(vocab_size, batch_size, 
                                                        number_batches,
                                                        data_iterator=train_iterator), 
                               model, indexer, train_iterator,
                               SimpleLossCompute(model.generator, 
                                                 criterion, 
                                                 optimizer),
                               print_every, data_group="Train")
        average_train_loss = train_loss.item()
        number_batches_processed += batch_size * number_batches
        if validate:
            print("\nRunning validation ...\n")
            model.eval() # evaluation model, gradient freezes.
            valid_losses = []
            valid_loss = run_epoch(get_linear_context_batch(vocab_size, 
                                                            batch_size, 
                                                            number_batches,
                                                            data_iterator=valid_iterator), 
                                   model, indexer, valid_iterator,
                                   SimpleLossCompute(model.generator, 
                                                     criterion, 
                                                     optimizer),
                                   print_every, data_group="Valid")  
            average_valid_loss = valid_loss.item()
            print("Train loss = %.5f" % (average_train_loss))
            print("Valid loss = %.5f\n" % (average_valid_loss))
            if average_valid_loss < best_valid_loss:
                print("Saving model weights for best valid: %.4f" % average_valid_loss)
                best_valid_loss = average_valid_loss
                save_path = model_dir+model_name+".ckpt"
                torch.save(model.state_dict(), save_path)
                print("Model saved to:", save_path, "\n")


class TransformerRunner(object):
    """Wrapper for loading and running trained Transformer."""
    
    def __init__(self, model_path, indexer_path, glove_path,
                 number_blocks, embed_size, upsample_size,
                 number_heads):
        """Initializer.
        
        Args:
            model_path: path to trained model (state dict).
            indexer_path: path to the pickled Indexer object (`.p` file).
            glove_path: path to the glove .txt file downloaded from
                        `https://nlp.stanford.edu/projects/glove/`.
            number_blocks: number of transformer encoder & decoder blocks.
            embed_size: embedding size.
            upsample_size: `blow-up` linear upsampling step (see Vaswani/17).
            number_heads: number of multi-attention heads.
        """
        self.indexer = dill.load(open(indexer_path, "rb"))
        vocab_size = self.indexer.size
        glove_init = load_glove(glove_path, self.indexer, embed_size)
        self.model = make_linear_context_model(vocab_size, vocab_size, 
                                               number_blocks, embed_size, 
                                               upsample_size, number_heads, 
                                               dropout_rate=0.0, glove_init=glove_init)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def to_inputs(self, sentence):
        tokens = [self.indexer.get_index(word, add=False) 
                  for word in sentence.split()]
        inputs = Variable(torch.LongTensor(np.array([tokens])))
        return inputs
        
    def paraphrase(self, sentence, context_sentences, 
                   max_length=20, pretty_print=True):
        """Paraphrase prediction with a given sentence (1 string).
        
        Args:
            sentence: a string.
            context_sentences: a list of string sentences.
            max_length: max decoding length.
            pretty_print: print out source and prediction in a given format.
        Returns:
            prediction: the predicted sentence as a string.
        """
        source = self.to_inputs(sentence).to(DEVICE)
        contexts = [self.to_inputs(context_sentence).to(DEVICE) 
                    for context_sentence in context_sentences]
        # Make mask: shape = <batch-size, 1, seq-length>
        source_mask = Variable(torch.ones(1, 1, source.size(1))).to(DEVICE)
        context_masks = [Variable(torch.ones(1, 1, context.size(1))).to(DEVICE)
                         for context in contexts]        
        source, source_mask = source.to(DEVICE), source_mask.to(DEVICE)
        prediction = linear_context_greedy_decode(self.model, source, contexts,
                                                  source_mask, context_masks, 
                                                  max_length, start_symbol=2) 
        source, prediction = [tensor_to_words(self.indexer, tensor) 
                              for tensor in [source.data, prediction]]
        source = " ".join([token for token in source 
                           if token not in ["PAD","<s>","</s>"]])
        prediction = " ".join([token for token in prediction 
                               if token not in ["PAD","<s>","</s>"]])
        if pretty_print:
            print("SOURCE     >> " + source + "\n" + \
                  "PREDICTION >> " + prediction + "\n")
        return prediction


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--indexer_dir", type=str)
    parser.add_argument("--glove_path", type=str)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--number_batches", type=int, default=1000)
    parser.add_argument("--number_epochs", type=int, default=10)
    parser.add_argument("--embed_size", type=int, default=300)
    parser.add_argument("--upsample_size", type=int, default=500)
    parser.add_argument("--number_heads", type=int, default=5)
    parser.add_argument("--number_blocks", type=int, default=4)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--saved_model", type=str)
    parser.add_argument("--model_name", type=str, default="transformer")
    parser.add_argument("--validate", type=int, default=1)
    args = parser.parse_args()
    
    
    train(data_dir=args.data_dir, 
          model_dir=args.model_dir, 
          indexer_dir=args.indexer_dir, 
          glove_path=args.glove_path,
          batch_size=args.batch_size, 
          number_batches=args.number_batches,
          number_epochs=args.number_epochs, 
          embed_size=args.embed_size,
          upsample_size=args.upsample_size,
          number_heads=args.number_heads,
          number_blocks=args.number_blocks,
          factor=args.factor,
          warmup=args.warmup, 
          learning_rate=args.learning_rate,
          dropout_rate=args.dropout_rate,
          print_every=args.print_every,
          saved_model=args.saved_model,
          model_name=args.model_name,
          validate=args.validate)
    