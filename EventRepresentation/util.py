"""Utility functions/classes for the Role-Factor Tensor Net. 

Author: Su Wang
"""

from collections import Counter
import linecache
import numpy as np


class Indexer(object):
    """Word<->Indexer bidirectional mapping."""
    
    def __init__(self):
        self.item_to_index = dict()
        self.index_to_item = dict()
        self.item_to_count = Counter()
        self.get_index("UNK", count=True)
        
    def __repr__(self):
        return "The size of the indexer = %d" % len(self.item_to_index)
    
    def __len__(self):
        return len(self.index_to_item)
    
    def get_item(self, index):
        if index in self.index_to_item:
            return self.index_to_item[index]
        return "UNK"
    
    def get_index(self, item, add=True, count=False):
        if count:
            self.item_to_count[item] += 1
        if item not in self.item_to_index:
            if add:
                index = len(self.item_to_index)
                self.item_to_index[item] = index
                self.index_to_item[index] = item
            else:
                return 0 # index of "UNK".
        return self.item_to_index[item] 
    
    def get_count(self, item):
        if item in self.item_to_count:
            return self.item_to_count[item]
        return 0
    
    def get_top_item_set(self, k):
        return set(item for item, count in self.item_to_count.most_common(k))
    
    def get_top_index_set(self, k):
        return set(self.get_index(item) 
                   for item, count in self.item_to_count.most_common(k))
    

def load_glove(glove_file_path, indexer, embedding_size=300):
    """Load pretrained GloVe embeddings (Pennington et al. 2014).
    
    Args:
        glove_file_path: path to the .txt glove embeddings.
        indexer: Indexer object.
        embedding_size: size of pretrained embeddings.
    Returns:
        <vocab_size, embedding_size> embedding matrix.
    """
    embeddings = np.zeros((len(indexer), embedding_size))
    wordset = set(indexer.item_to_index.keys())
    number_loaded = 0
    print("Loading glove embeddings (size = %d)\n" % embedding_size)
    with open(glove_file_path, 'r') as glove_file:
        for i, line in enumerate(glove_file):
            line = line.split()
            word, embedding = line[0], np.array(line[1:], dtype=float)
            if word in wordset:
                word_index = indexer.get_index(word, add=False, count=False)
                embeddings[word_index] = embedding
                number_loaded += 1
            if i != 0 and i % 100000 == 0:
                print('... processed %d lines.' % i)
    print('\nDone!\n')
    print('Loaded %d | OOV size = %d\n' % (number_loaded,
                                           len(indexer)-number_loaded))
    return embeddings
    

def read_event_file(indexer, event_file_path):
    """Index tokens and get document segments.
    
    Args:
        indexer: Index object.
        event_file_path: each line has <subject, predicate, dobject>, separator = `\t`.
    Returns:
        document_bounds: a list of (start-line-number, end-line-number) document bounds.
    """
    document_bounds = []
    with open(event_file_path, "r") as event_file:
        left_bound = 1
        for i, line in enumerate(event_file):
            if line == "\n":
                right_bound = i + 1 # python indexes from 0, linecase from 1.
                if left_bound < right_bound:
                    document_bounds.append((left_bound, right_bound))
                left_bound = right_bound + 1
            else:
                [indexer.get_index(item, count=True) 
                 for item in line.strip().split("\t")]
    return document_bounds


class EventIterator(object):
    """Iterator over a single event document file."""
    
    def __init__(self, indexer, document_bounds, event_file_path):
        """Initializer.
        
        Args:
            indexer: Indexer object.
            document_bounds: a list of (start-line-number, end-line-number) document bounds.
            event_file_path: each line has <subject, predicate, dobject>, separator = `\t`.
        """
        self.indexer = indexer
        self.document_bounds = document_bounds
        self.event_file_path = event_file_path
        self.current_document_id = 0
        self.current_epoch = 0
    
    def read_line(self, line_id):
        """Read an event from a line number, returns [subject_id, predicate_id, dobject_id]."""
        return [self.indexer.get_index(item, count=False) for item 
                in linecache.getline(self.event_file_path, line_id).strip().split("\t")]
    
    def sample_target_ids_and_events(self, document_bound, sample_size=5):
        """Sample `sample_size` of target events given a document."""
        left_bound, right_bound = document_bound
        sample_line_ids = np.random.choice(range(left_bound, right_bound),
                                           size=sample_size)
        sample_events = np.array([self.read_line(line_id) for line_id in sample_line_ids])
        return sample_line_ids, sample_events
    
    def sample_positive_events(self, document_bound, target_line_id, 
                               window=5, sample_size=5):
        """Sample `sample_size` events within `window` distance from a target event."""
        left_bound, right_bound = document_bound
        left_window_bound = max(left_bound, target_line_id-window)
        right_window_bound = min(right_bound, target_line_id+window)
        sample_line_ids = np.random.choice(range(left_window_bound, right_window_bound),
                                           size=sample_size)
        sample_events = np.array([self.read_line(line_id) for line_id in sample_line_ids])
        return sample_events
    
    def sample_negative_events(self, sample_size=5):
        """Sample `sample_size` events randomly from the entire document file."""
        sample_events = []
        while len(sample_events) < sample_size:
            line_id = np.random.choice(range(1, self.document_bounds[-1][1]))
            sample_event = self.read_line(line_id)
            if len(sample_event) == 1: continue # hit document separator
            sample_events.append(sample_event)
        return np.array(sample_events)
            
    def get_batch(self, target_size=5, window=5, sample_size=5):
        """Return a batch of (target, positive, negative) events."""
        if self.current_document_id >= len(self.document_bounds):
            self.current_document_id = 0
            self.current_epoch += 1
        document_bound = self.document_bounds[self.current_document_id]
        self.current_document_id += 1
        target_line_ids, target_events = self.sample_target_ids_and_events(document_bound,
                                                                           target_size)
        batch_events = batch_positive = batch_negative = np.array([[0, 0, 0]])
        for target_line_id, target_event in zip(target_line_ids, target_events):
            batch_events = np.vstack((batch_events, 
                                      np.array([target_event 
                                                for _ in range(sample_size)])))
            batch_positive = np.vstack((batch_positive, 
                                        self.sample_positive_events(document_bound,
                                                                    target_line_id,
                                                                    window,
                                                                    sample_size)))
            batch_negative = np.vstack((batch_negative,
                                        self.sample_negative_events(sample_size)))
        # Drop the initializing empty event.
        return batch_events[1:], batch_positive[1:], batch_negative[1:]
        