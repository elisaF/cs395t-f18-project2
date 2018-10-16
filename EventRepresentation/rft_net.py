"""Implementing Role-Factor Tensor Net (Weber et al. 2018)

Author: Su Wang
"""

from __future__ import division
from __future__ import print_function
from copy import deepcopy
import dill
import math
import numpy as np
import os
import time
import torch
from torch.autograd import Variable as Var
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from util import *

CUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Embeddings(nn.Module):
    """Embedding lookup table."""
    
    def __init__(self, vocab_size, embedding_size, glove_init=None):
        """Initializer.
        
        Args:
            vocab_size: size of input vocabulary (max index - 1).
            embedding_size: embedding size.
            glove_init: None or <vocab_size, embedding_size> embedding matrix.
        """
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        if glove_init is not None:
            assert glove_init.shape == (vocab_size, embedding_size)
            self.embed.weight.data.copy_(torch.from_numpy(glove_init))
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
    
    def info(self):
        """Print out embedding lookup table stats."""
        print("Embedding lookup table of size <%d, %d>" % (self.vocab_size,
                                                           self.embedding_size))
    
    def forward(self, batch_inputs):
        """Forward pass.
        
        Args:
            batch_inputs: Variable.Tensor of the shape 
                          <batch_size, ..., sequence length>.
        Returns:
            Variable.Tensor of the shape <batch_size, ..., sequence length, embedding_size>.
        """
        return self.embed(batch_inputs) * math.sqrt(self.embedding_size)


class RoleFactorTensorNet(nn.Module):
    """Role-Factor Tensor Net (Weber et al. 2018)."""
    
    def __init__(self, embedding_size, hidden_size, output_size):
        """Initializer.
        
        Args:
            embedding_size: word embedding size.
            hidden_size: model hidden size.
            output_size: event embedding size.
        """
        super(RoleFactorTensorNet, self).__init__()
        self.T = torch.FloatTensor(hidden_size, embedding_size, embedding_size).to(CUDA)
        nn.init.xavier_uniform_(self.T)
        self.W1 = nn.Linear(hidden_size, output_size)
        self.W2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, batch_subjects, batch_predicates, batch_dobjects):
        """Forward pass.
        
        Args:
            batch_subjects: batch of subject embeddings, 
                            shape <batch_size, embedding_size>.
            batch_predicates: same as batch_subjects, for predicates.
            batch_dobjects: same as batch_subjects, for direct objects.
        Returns:
            Batch of event embeddings, shape <batch_size, output_size>.
        """
        # Einstein sum for tensor contraction (Weber/18, Eq.6).
        batch_v_subjects = torch.einsum("ijk,bj,bk->bi", [self.T, 
                                                          batch_subjects,
                                                          batch_predicates])
        batch_v_dobjects = torch.einsum("ijk,bj,bk->bi", [self.T, 
                                                          batch_dobjects,
                                                          batch_predicates.clone()])
        # Argument composition through linear layer (Weber/18, Eq.7).
        return self.W1(batch_v_subjects) + self.W2(batch_v_dobjects)


def compose_event(batch_inputs, embedder, rft_net):
    """Input->Event composition with Role-Factor Tensor Net.
    
    Args:
        batch_inputs: numpy ndarray, shape = <batch_size, svo=3>.
        embedder: Embeddings object.
        rft_net: RoleFactorTensorNet object.
    Returns:
        Compositional event embedding, shape = <batch_size, event_size>.
    """
    batch_subjects = embedder(torch.LongTensor(batch_inputs[:, 0]).to(CUDA))
    batch_predicates = embedder(torch.LongTensor(batch_inputs[:, 1]).to(CUDA))
    batch_dobjects = embedder(torch.LongTensor(batch_inputs[:, 2]).to(CUDA))
    return rft_net(batch_subjects, batch_predicates, batch_dobjects)


def event_to_integer(indexer, event):
    return np.array([indexer.get_index(word, add=False, count=False) 
                     for word in event])


def compute_event_similarity(event1, event2, indexer, embedder, rft_net):
    """Compute similarity for a pair of events with trained RFT-Net.
    
    Args:
        event1: (subject, predicate, dobject) string tuple.
        event2: same as event1.
        indexer: Indexer object.
        embedder: Embeddings object.
        rft_net: RoleFactorTensorNet object.
    Returns:
        Cosine similarity between event1 and event2 (as composed with RFT-Net).
    """
    event1 = event_to_integer(indexer, event1)
    event2 = event_to_integer(indexer, event2)
    event1_embedding = compose_event(np.array([event1]), embedder, rft_net)
    event2_embedding = compose_event(np.array([event2]), embedder, rft_net)
    return F.cosine_similarity(event1_embedding, event2_embedding).item()


def run_batch(batch, embedder, rft_net, optimizer, margin, batch_size, data_group):
    """Run training routine on a single batch.
    
    Args: 
        batch: the data tuple that contains inputs, positive and negative targets.
               They are all numpy ndarray, shape = <batch_size, svo=3>.
        embedder: Embeddings object.
        rft_net: RoleFactorTensorNet object.
        optimizer: torch.nn.optim.* optimizer.
        margin: float margin hyperparameter for hinge loss.
        batch_size: int, for computing batch average loss.
        data_group: `train` or `valid`.
    Returns:
        Batch loss.
    """
    if data_group == "train":
        rft_net.train()
        embedder.train()
    elif data_group == "valid":
        rft_net.eval()
        embedder.eval()
    else:
        raise ValueError("Parameter `data_group` must be either `train` or `valid`!")
    batch_events, batch_positive, batch_negative = batch
    # Encode svo-triples as event vectors,
    #   shape = <batch_size, event_size>.
    batch_events = compose_event(batch_events, embedder, rft_net)
    batch_positive = compose_event(batch_positive, embedder, rft_net)
    batch_negative = compose_event(batch_negative, embedder, rft_net)    
    # Compute batch similarity, shape = <batch_size, >.
    similarity_positive = F.cosine_similarity(batch_events, batch_positive)
    similarity_negative = F.cosine_similarity(batch_events, batch_negative)
    similarity_difference = torch.mean(similarity_negative) - torch.mean(similarity_positive)
    # Compute Hinge Loss: mean(sum(max(0.0, margin + sim_neg - sim_pos))).
    #   Weber/18, page 3.
    loss = torch.max(torch.FloatTensor(np.array(0.0)).to(CUDA), 
                     margin + similarity_difference)
    if data_group == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item() / batch_size


def train(train_event_file_path, valid_event_file_path, glove_init,
          indexer, train_document_bounds, valid_document_bounds,
          embedding_size, hidden_size, output_size,
          number_epochs, target_size, window, sample_size,
          learning_rate, margin,
          print_every,
          trained_rft=None, trained_embedder=None,
          rft_save_file=None, embedder_save_file=None):
    """Trainer for Role-Factor Tensor Net (RFT).
    
    Args:
        train_event_file_path: each line has <subject, predicate, dobject>, separator = `\t`.
        valid_event_file_path: same as `train_event_file_path`, but for validation.
        glove_init: None or <vocab_size, embedding_size> glove embeddings.
        indexer: Indexer object.
        train_document_bounds: a list of (start-line-number, end-line-number) document bounds.
        valid_document_bounds: same as `train_document_bounds`, but for validation.
        embedding_size: word embedding size.
        hidden_size: model hidden size for the RFT net.
        output_size: event embedding size.
        number_epochs: number of epochs for each event file.
        target_size: sample size of target events.
        window: window size for positive neighbor events for a target event.
        sample_size: sample size of positive/negative events for a target event.
        learning_rate: learning rate.
        margin: margin for the Hinge Loss.
        print_every: printout report after every `print_every` documents.
        trained_rft: path to a trained RFT model.
        trained_embedder: path to a trained word embedder.
        rft_save_file: path where the RFT model is saved.
        embedder_save_file: path where the word embedder is saved.
    Returns:
        rft_net: a trained RFT net.
        embedder: a trained word embedding object.
    """
    
    batch_size = target_size * sample_size # later for computing batch average loss.
    rft_net = RoleFactorTensorNet(embedding_size, hidden_size, output_size).to(CUDA)
    if trained_rft is not None:
        rft_net.load_state_dict(torch.load(trained_rft))
        print("Loaded pretrained RFT net.\n")
    embedder = Embeddings(vocab_size=len(indexer),
                          embedding_size=embedding_size,
                          glove_init=glove_init).to(CUDA)
    if trained_embedder is not None:
        embedder.load_state_dict(torch.load(trained_embedder))
        print("Loaded pretrained word embedder.\n")
    
    optimizer = optim.Adam(rft_net.parameters(), lr=learning_rate)
    
    global_step = 0
    best_valid_loss = np.inf
    train_iterator = EventIterator(indexer, train_document_bounds, 
                                   train_event_file_path)
    train_losses = []
    while train_iterator.current_epoch < number_epochs:
        # print("Epoch %d:\n" % (train_iterator.current_epoch+1))
        global_step += 1
        # Get batch inputs, shape = <batch_size, svo=3>.
        batch = train_iterator.get_batch(target_size, window, sample_size)
        loss = run_batch(batch, embedder, rft_net, optimizer, margin,
                         batch_size, data_group="train")
        train_losses.append(loss)
        if global_step % print_every == 0:
            valid_iterator = EventIterator(indexer, valid_document_bounds, 
                                           valid_event_file_path)
            valid_losses = []
            while valid_iterator.current_epoch < 1:
                batch = valid_iterator.get_batch(target_size, window, sample_size)
                loss = run_batch(batch, embedder, rft_net, optimizer, margin,
                                 batch_size, data_group="valid")
                valid_losses.append(loss)
            average_train_loss = np.mean(train_losses)
            average_valid_loss = np.mean(valid_losses)
            print("@Step-%d:" % global_step)
            print("  Train loss = %.4f" % average_train_loss)
            print("  Validation loss = %.4f\n" % average_valid_loss)
            if average_valid_loss < best_valid_loss:
                print("Saving model weights for best valid: %.4f" % average_valid_loss)
                best_valid_loss = average_valid_loss
                torch.save(rft_net.state_dict(), rft_save_file)
                torch.save(embedder.state_dict(), embedder_save_file)
                print("RFT net saved to:", rft_save_file)
                print("Word embedder saved to:", embedder_save_file, "\n")
    print("\n")

    return rft_net, embedder 


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_event_file_path", type=str)
    parser.add_argument("--valid_event_file_path", type=str)
    parser.add_argument("--glove_file_path", type=str)
    parser.add_argument("--embedding_size", type=int)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--output_size", type=int)
    parser.add_argument("--number_epochs", type=int)
    parser.add_argument("--target_size", type=int)
    parser.add_argument("--window", type=int)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--margin", type=float)
    parser.add_argument("--print_every", type=int)
    parser.add_argument("--trained_rft_file", type=str)
    parser.add_argument("--trained_embedder_file", type=str)
    parser.add_argument("--rft_save_file", type=str)
    parser.add_argument("--embedder_save_file", type=str)
    parser.add_argument("--indexer_save_file", type=str)
    args = parser.parse_args()    
    
    indexer = Indexer()
    train_document_bounds = read_event_file(indexer, args.train_event_file_path)
    valid_document_bounds = read_event_file(indexer, args.train_event_file_path)
    dill.dump(indexer, open(args.indexer_save_file, "wb"))
    print("Word indexer prepared and saved to:", args.indexer_save_file)
    glove_init = load_glove(args.glove_file_path, indexer, args.embedding_size)
    rft_net, embedder = train(train_event_file_path=args.train_event_file_path, 
                              valid_event_file_path=args.valid_event_file_path,
                              glove_init=glove_init,
                              indexer=indexer, 
                              train_document_bounds=train_document_bounds,
                              valid_document_bounds=valid_document_bounds,
                              embedding_size=args.embedding_size,
                              hidden_size=args.hidden_size,
                              output_size=args.output_size,
                              number_epochs=args.number_epochs,
                              target_size=args.target_size,
                              window=args.window,
                              sample_size=args.sample_size,
                              learning_rate=args.learning_rate,
                              margin=args.margin,
                              print_every=args.print_every,
                              trained_rft=args.trained_rft_file,
                              trained_embedder=args.trained_embedder_file,
                              rft_save_file=args.rft_save_file,
                              embedder_save_file=args.embedder_save_file)
    
    