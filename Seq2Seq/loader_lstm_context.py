"""Data preprocessing utils for LSTM-context variant of Transformer.

Author: Su Wang
"""

import numpy as np
import os
import torch
from torch.autograd import Variable


class Indexer(object):
    """Word to index bidirectional mapping."""
    
    def __init__(self, start_symbol="<s>", end_symbol="</s>"):
        """Initializing dictionaries and (hard coded) special symbols."""
        self.word_to_index = {}
        self.index_to_word = {}
        self.size = 0
        # Hard-code special symbols.
        self.get_index("PAD", add=True) 
        self.get_index("UNK", add=True) 
        self.get_index(start_symbol, add=True)
        self.get_index(end_symbol, add=True)
    
    def __repr__(self):
        """Print size info."""
        return "This indexer currently has %d words" % self.size
    
    def get_word(self, index):   
        """Get word by index if its in range. Otherwise return `UNK`."""
        return self.index_to_word[index] if index < self.size and index >= 0 else "UNK"

    def get_index(self, word, add):
        """Get index by word. If `add` is on, also append word to the dictionaries."""
        if self.contains(word):
            return self.word_to_index[word]
        elif add:
            self.word_to_index[word] = self.size
            self.index_to_word[self.size] = word
            self.size += 1
            return self.word_to_index[word]
        return self.word_to_index["UNK"]
        
    def contains(self, word):
        """Return True/False to indicate whether a word is in the dictionaries."""
        return word in self.word_to_index
    
    def add_sentence(self, sentence, add):
        """Add all the words in a sentence (a string) to the dictionary."""
        indices = [self.get_index(word, add) for word in sentence.split()]
        return indices

    def add_document(self, document_path, add):
        """Add all the words in a document (a path to a text file) to the dictionary."""
        indices_list = []
        with open(document_path, "r") as document:
            for line in document:
                indices = self.add_sentence(line, add)
                indices_list.append(indices)
        return indices_list
    
    def to_words(self, indices):
        """Indices (ints) -> words (strings) conversion."""
        return [self.get_word(index) for index in indices]
    
    def to_sent(self, indices):
        """Indices (ints) -> sentence (1 string) conversion."""
        return " ".join(self.to_words(indices))
    
    def to_indices(self, words):
        """Words (strings) -> indices (ints) conversion."""
        return [self.get_index(word) for word in words]


def load_document(document_path, indexer, add):
    """Load a document as a list of indices where each corresponds to a sentence.
    
    Args:
        document_path: path to a text file where each line has 1 sentence.
        indexer: Indexer object.
        add: boolean. True if add words to dictionaries in `indexer` while indexing.
    Returns:
        A list of lists of indices. Each sublist is a sentence.
    """
    assert os.path.exists(document_path)
    print("... loading `" + document_path + "`")
    indices_list = indexer.add_document(document_path, add)
    return indices_list


def load_corpus(data_dir):
    """Load a corpus with specified format (see below).
    
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
    Returns:
        indexer: Indexer object.
        train_source: a list of lists of indices (sublist: a sentence).
        train_context1-4: same as above.
        train_target: same as above.
        valid_source: same as above.
        valid_context1-4: same as above.
        valid_target: same as above.
    """
    source_files = ["train_context1.txt", "train_context2.txt", 
                    "train_source.txt", "train_target.txt",
                    "valid_context1.txt", "valid_context2.txt",
                    "valid_source.txt", "valid_target.txt"]
    assert all(source_file in set(os.listdir(data_dir)) for source_file in source_files)
    indexer = Indexer()
    train_source = load_document(data_dir+"train_source.txt", indexer, add=True)
    train_context1 = load_document(data_dir+"train_context1.txt", indexer, add=True)
    train_context2 = load_document(data_dir+"train_context2.txt", indexer, add=True)
    train_context3 = load_document(data_dir+"train_context3.txt", indexer, add=True)
    train_context4 = load_document(data_dir+"train_context4.txt", indexer, add=True)
    train_target = load_document(data_dir+"train_target.txt", indexer, add=True)
    valid_source = load_document(data_dir+"valid_source.txt", indexer, add=False)
    valid_context1 = load_document(data_dir+"valid_context1.txt", indexer, add=False)
    valid_context2 = load_document(data_dir+"valid_context2.txt", indexer, add=False)
    valid_context3 = load_document(data_dir+"valid_context3.txt", indexer, add=False)
    valid_context4 = load_document(data_dir+"valid_context4.txt", indexer, add=False)
    valid_target = load_document(data_dir+"valid_target.txt", indexer, add=False)
    return indexer, \
           train_source, \
           train_context1, train_context2, train_context3, train_context4, \
           train_target, \
           valid_source, \
           valid_context1, valid_context2, valid_context3, valid_context4, \
           valid_target


def load_glove(glove_path, indexer, embed_size=300):
    """Load pretrained GloVe embeddings (Pennington et al. 2014).
    
    Args:
        glove_path: path to the glove .txt file downloaded from
                    `https://nlp.stanford.edu/projects/glove/`.
        indexer: Indexer object. Words are added by reading data by now.
        embed_size: embedding size.
    Returns:
        embeddings: a numpy ndarray of shape <vocab-size, embed-size>.
    """
    embeddings = np.zeros((indexer.size, embed_size))
    number_loaded = 0
    print("Loading glove embeddings (size = %d)\n" % embed_size)
    with open(glove_path, "r") as glove_file:
        for i, line in enumerate(glove_file):
            line = line.split()
            word, embedding = line[0], np.array(line[1:], dtype=float)
            if indexer.contains(word):
                word_index = indexer.get_index(word, add=False)
                embeddings[word_index] = embedding
                number_loaded += 1
            if i != 0 and i % 100000 == 0:
                print("... processed %d lines." % i)
    print("\nDone!\n")
    print("Loaded %d | OOV size = %d\n" % (number_loaded, indexer.size-number_loaded))
    return embeddings


class DataIterator:
    """Iterator over train or validation data."""
    
    def __init__(self, source, 
                 context1, context2, context3, context4,
                 target, size):
        """Initializer.
        
        Args:
            source: a list of lists of indices (sublist: a sentence).
            context1-4: same as above.
            target: same as above.
            size: number of train/valid entries (for quick lookup).
        """
        self.source = source
        self.context1 = context1
        self.context2 = context2
        self.context3 = context3
        self.context4 = context4
        self.target = target
        self.size = size
        self.indices = range(self.size)
        self.pad_index = 0
        self.start_symbol_index = 2 # hard-coded
        self.end_symbol_index = 3 # hard-coded
        
    def _pad_sentence(self, index, max_source_length, max_context_length, max_target_length):
        """Sentence padder.
        
        Pad sentences to given max lengths by first adding start & end symbol then
        adding `pad_index` to the end until length met.
        
        Args:
            index: integer index to retrieve sentence entry (list of indices).
            max_source_length: max length of source sequence.
            max_context_length: max *total* lenght of context sequences (apply to all).
            max_target_length: max length of target sequence.
        Returns:
            source_padded: list of indices for a sentence, padded to given max length.
            context1-4_padded: same as above.
            target_padded: same as above.
            source_length: length of the source sentence.
        """
        source_tokens = [self.start_symbol_index] + self.source[index] + [self.end_symbol_index]
        context1_tokens = [self.start_symbol_index] + self.context1[index] + [self.end_symbol_index]
        context2_tokens = [self.start_symbol_index] + self.context2[index] + [self.end_symbol_index]
        context3_tokens = [self.start_symbol_index] + self.context3[index] + [self.end_symbol_index]
        context4_tokens = [self.start_symbol_index] + self.context4[index] + [self.end_symbol_index]
        context_tokens = context1_tokens + context2_tokens + context3_tokens + context4_tokens
        target_tokens = [self.start_symbol_index] + self.target[index] + [self.end_symbol_index]
        source_length = len(source_tokens)
        context_length = len(context_tokens)
        target_length = len(target_tokens)
        source_padded = source_tokens[:max_source_length-1]+[self.end_symbol_index] \
            if source_length > max_source_length \
            else source_tokens+[self.pad_index]*(max_source_length-source_length)      
        context_padded = context_tokens[:max_context_length-1]+[self.end_symbol_index] \
            if context_length > max_context_length \
            else context_tokens+[self.pad_index]*(max_context_length-context_length)
        target_padded = target_tokens[:max_target_length-1]+[self.end_symbol_index] \
            if target_length > max_target_length \
            else target_tokens+[self.pad_index]*(max_target_length-target_length)
        return source_padded, context_padded, \
               target_padded, source_length
    
    def random_batch(self, batch_size, 
                     max_source_length=20, max_context_length=80, max_target_length=20):
        """Randomly retrieve a batch of source-target sentence pairs in torch tensors.
        
        Args:
            batch_size: batch size.
            max_source_length: max length of source sequence.
            max_context_length: max lenght of context sequences (apply to all).
            max_target_length: max length of target sequence.
        Returns:
            batch_source: <batch-size, max-source-length> shaped source sentences.
            batch_target: <batch-size, max-target-length> shaped target sentences.
        """
        batch_indices = np.random.choice(self.indices, size=batch_size, replace=False)
        batch_source, batch_context, batch_target = [], [], []
        batch_source_length = []
        for index in batch_indices:
            source_padded, context_padded, \
            target_padded, source_length = self._pad_sentence(index, max_source_length,
                                                                     max_context_length,
                                                                     max_target_length)
            batch_source.append(source_padded)
            batch_context.append(context_padded)
            batch_target.append(target_padded)
            batch_source_length.append(source_length)
        # Sort batch items by the length of the source in descending (pytorch quirk).
        batch_indices = [index for index, length in sorted(zip(range(batch_size), 
                                                               batch_source_length),
                                                           key=lambda k:k[1],
                                                           reverse=True)]
        # Batch-major: <batch-size, seq-length>
        batch_source = Variable(torch.LongTensor(np.array(batch_source)[batch_indices]))
        batch_context = Variable(torch.LongTensor(np.array(batch_context)[batch_indices]))
        batch_target = Variable(torch.LongTensor(np.array(batch_target)[batch_indices]))
        return batch_source, batch_context, batch_target
