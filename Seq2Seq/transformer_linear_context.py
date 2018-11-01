# transformer_linear_context.py

import copy
import dill
import math
import numpy as np
import os
import random
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearContextEncoderDecoder(nn.Module):
    """Standard Transformer encoder with context encoding.
    
    The context encoding treats every context sentence the same way as it
    does with the source sentence. The multiple encodings are finally
    concatenated and projected to the size of that of the source sentence
    on its own.
    """
    
    def __init__(self, encoder, decoder, 
                 source_embed, target_embed, 
                 generator, 
                 embed_size, context_size):
        """
        Args:
            encoder: Encoder object.
            decoder: Decoder object.
            source_embed: Sequential(Embedding, PositionalEncoding) pipeline for source sequence.
                          The embedder is shared with contexts.
            target_embed: same as `source_embed`, but for target sequence.
            generator: Generator object, final linear-softmax layer.
            embed_size: embedding size.
            context_size: number of context inputs.
        """
        super(LinearContextEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embed = source_embed
        self.target_embed = target_embed
        self.generator = generator
        self.context_size = context_size
        self.linear = nn.Linear(embed_size + embed_size * self.context_size,
                                embed_size)
        
    def forward(self, source, contexts, target, 
                source_mask, context_masks, target_mask):
        """Take in and process masked src and target sequences.
        
        Args:
            source: torch.LongTensor object, shape = <batch-size, encoder-seq-length>.
            contexts: a list of torch.LongTensor objects, each is of the same shape as `source`.
            target: same as `source`, shape = <batch-size, decoder-seq-length>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
            context_masks: a list of torch.LongTensor, each has the same shape as `source_mask`.
            target_mask: same `source_mask`, shape = <batch-size, 1, decoder-seq-length>.
        Returns:
            decoder outputs of the shape <batch-size, decoder-seq-length, embed-size>.
        """
        encoded_source = self.encode(source, source_mask)
        encoded_contexts = []
        for context, context_mask in zip(contexts, context_masks):
            encoded_contexts.append(self.encode(context, context_mask))
        encoded = self.linear(torch.cat([encoded_source] + encoded_contexts, dim=-1))
        return self.decode(encoded, 
                           source_mask, target, target_mask)

    def encode(self, source, source_mask):
        """
        Args:
            source: torch.LongTensor object, shape = <batch-size, encoder-seq-length>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
        Returns:
            Encoded hidden states of the shape <batch-size, encoder-seq-length, embed-size>.
        """
        memory = self.encoder(self.source_embed(source), source_mask)
        return memory
               
    
    def decode(self, memory, source_mask, target, target_mask):
        """
        Args:
            memory: torch.LongTensor, shape = <batch-size, encoder-seq-length, embed-size>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
            target: same as source, shape = <batch-size, decoder-seq-length>.
            target_mask: same `source_mask`, shape = <batch-size, 1, decoder-seq-length>.
        Returns:
            Decoded hidden states of the shape <batch-size, decoder-seq-length, embed-size>.
        """
        return self.decoder(self.target_embed(target), memory, source_mask, target_mask)
    
    
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    
    def __init__(self, embed_size, vocab_size):
        """
        Args:
            embed_size: embedding size.
            vocab_size: vocab size (of the decoder).
        """
        super(Generator, self).__init__()
        self.final_linear = nn.Linear(embed_size, vocab_size)

    def forward(self, decoder_outputs):
        """
        Args:
            decoder_outputs: shape = <batch-size, decoder-seq-length, embed-size>.
        Returns:
            Prediction of the shape <batch-size, decoder-seq-length, vocab-size>.
        """
        return F.log_softmax(self.final_linear(decoder_outputs), dim=-1)


def clones(module_block, number_blocks):
    """Produce copies of identical module blocks.
    
    Args:
        module_block: any nn.Module object.
        number_blocks: number of copies.
    Returns:
        A nn.ModuleList with `number_blocks` copies of `module_block`.
    """
    return nn.ModuleList([copy.deepcopy(module_block) for _ in range(number_blocks)])


class Encoder(nn.Module):
    """Core encoder is a stack of module blocks (Fig 1, Vaswani/17)."""
    
    def __init__(self, block, number_blocks):
        """
        Args:
            block: EncoderBlock
            number_blocks: number of blocks.
        """
        super(Encoder, self).__init__()
        self.blocks = clones(block, number_blocks)
        self.norm = LayerNorm(block.number_features)
        
    def forward(self, source_inputs, source_mask):
        """Pass the input (and mask) through each block in turn.
        
        Args:
            source_inputs: embedded source, shape = <batch-size, encoder-seq-length, embed-size>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
        Returns:
            Encoded hidden states of the shape <batch-size, encoder-seq-length, embed-size>.
        """
        for block in self.blocks:
            source_inputs = block(source_inputs, source_mask) # shape does not change.
        return self.norm(source_inputs) # apply layer normalization (Ba/16).
    

class LayerNorm(nn.Module):
    """Construct a layernorm module (notation see Ba/16)."""
    
    def __init__(self, number_features, eps=1e-6):
        """
        Args:
            number_features: hidden size of the inputs to be layer-normalized.
            eps: perturbation size.
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(number_features))
        self.b_2 = nn.Parameter(torch.zeros(number_features))
        self.eps = eps

    def forward(self, inputs):
        """
        Args:
            inputs: shape = <batch-size, encoder-seq-length, embed-size>.
        Returns:
            Layer-normalized inputs, same shape.
        """
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self.a_2 * (inputs - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    
    def __init__(self, number_features, dropout_rate):
        """
        Args:
            number_features: hidden size of the inputs.
            dropout_rate: dropout rate.
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(number_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        """Apply residual connection to any sublayer with the same size.
        
        Args:
            inputs: shape = <batch-size, encoder-seq-length, embed-size>.
            sublayer: TODO.
        Returns:
            Processed inputs, same shape.
        """
        return inputs + self.dropout(sublayer(self.norm(inputs)))


class EncoderBlock(nn.Module):
    """Encoder is made up of multi-headed attention and feed forward."""
    
    def __init__(self, number_features, multi_self_attention, feed_forward, dropout_rate):
        """
        Args:
            number_features: hidden size of the inputs.
            multi_self_attention: MultiHeadAttention object.
            feed_forward: PointwiseFeedForward object.
            dropout_rate: dropout rate.
        """
        super(EncoderBlock, self).__init__()
        self.multi_self_attention = multi_self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(number_features, dropout_rate), 
                               number_blocks=2) # 2: see Vaswani/17.
        self.number_features = number_features

    def forward(self, source_inputs, source_mask):
        """
        Args:
            source_inputs: embedded source, shape = <batch-size, encoder-seq-length, embed-size>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
        Returns:
            Encoded hidden states of the shape <batch-size, encoder-seq-length, embed-size>.
        """        
        # Apply self attention with multi-headed attention at the first layer in the block.
        # Apply feedforward at the second layer in the block.
        # Shape does not change in the process.
        source_inputs = self.sublayer[0](source_inputs, 
                                         lambda inputs: self.multi_self_attention(inputs, 
                                                                                  inputs,
                                                                                  inputs, 
                                                                                  source_mask))
        return self.sublayer[1](source_inputs, self.feed_forward)
    

class Decoder(nn.Module):
    """Generic block decoder with masking."""
    
    def __init__(self, block, number_blocks):
        """
        Args:
            block: DecoderBlock
            number_blocks: number of blocks.
        """        
        super(Decoder, self).__init__()
        self.blocks = clones(block, number_blocks)
        self.norm = LayerNorm(block.number_features)
        
    def forward(self, target_inputs, memory, source_mask, target_mask):
        """
        Args:
            target_inputs: <batch-size, decoder-seq-length, embed-size>
            memory: torch.LongTensor, shape = <batch-size, encoder-seq-length, embed-size>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
            target_mask: same `source_mask`, shape = <batch-size, 1, decoder-seq-length>.
        Returns:
            Decoder hidden states of the shape <batch-size, decoder-seq-length, embed-size>.
        """
        for block in self.blocks:
            target_inputs = block(target_inputs, memory, source_mask, target_mask)
        return self.norm(target_inputs)
    
    
class DecoderBlock(nn.Module):
    """Decoder is made of self attention, source attention, and feed forward."""
    
    def __init__(self, number_features, 
                 multi_self_attention, multi_source_attention, 
                 feed_forward, dropout_rate):
        """
        Args: 
            number_features: hidden size of the inputs.
            multi_self_attention: MultiHeadAttention object.
            multi_source_attention: MultiHeadAttention object.
            feed_forward: PointwiseFeedForward object.
            dropout_rate: dropout rate.
        """
        super(DecoderBlock, self).__init__()
        self.number_features = number_features
        self.multi_self_attention = multi_self_attention
        self.multi_source_attention = multi_source_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(number_features, dropout_rate), 
                               number_blocks=3) # 3: see Vaswani/17.
 
    def forward(self, target_inputs, memory, source_mask, target_mask):
        """
        Args:
            target_inputs: <batch-size, decoder-seq-length, embed-size>
            memory: torch.LongTensor, shape = <batch-size, encoder-seq-length, embed-size>.
            source_mask: torch.LongTensor, shape = <batch-size, 1, encoder-seq-length>.
            target_mask: same `source_mask`, shape = <batch-size, 1, decoder-seq-length>.
        Returns:
            Decoder hidden states of the shape <batch-size, decoder-seq-length, embed-size>.
        """
        # Apply self attention with multi-headed attention at the first layer in the block.
        # Apply source attention with multi-headed attention at the second layer.
        # Apply feedforward at the third layer.
        # Shape does not change in the process.
        target_inputs = self.sublayer[0](target_inputs, 
                                         lambda inputs: self.multi_self_attention(inputs, 
                                                                                  inputs, 
                                                                                  inputs, 
                                                                                  target_mask))
        target_inputs = self.sublayer[1](target_inputs, 
                                         lambda inputs: self.multi_source_attention(inputs, 
                                                                                    memory, 
                                                                                    memory, 
                                                                                    source_mask))
        return self.sublayer[2](target_inputs, self.feed_forward)


def subsequent_mask(sequence_length):
    """Mask out subsequent positions."""
    # Make matrix of the shape, e.g.:
    #   array([[0, 1, 1],
    #          [0, 0, 1],
    #          [0, 0, 0]])
    attention_shape = (1, sequence_length, sequence_length)
    subsequent_mask = np.triu(np.ones(attention_shape), k=1).astype("uint8")
    # Make torch tensor of the shape, e.g.:
    #   1  0  0
    #   1  1  0
    #   1  1  1
    #   type = torch.ByteTensor.
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention.
    
    Args:
        query: shape = <batch-size, number-heads, seq-length, embed-size/number-heads>.
               where attention-size = embed-size/number-heads.
        key: same shape as `query`.
        value: same shape as `query`.
        mask: <batch-size, 1, 1, seq-length>.
    Returns:
        Attention scores: <batch-size, number-heads, seq-length, embed-size/number-heads>.
        Attention weights: <batch-size, number-heads, seq-length, seq-length>
    """
    attention_size = query.size(-1) # embed-size/number-heads
    # Operation 1. transpose: <batch-size, number-heads, seq-length, attention-size> 
    #                      -> <batch-size, number-heads, attention-size, seq-length>
    # Operation 2. matmul: <batch-size, number-heads, seq-length, attention-size> 
    #                    * <batch-size, number-heads, attention-size, seq-length>
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(attention_size)

    if mask is not None:
        # Fill the cells in scores corresponding to mask where the mask cell == 0.
        scores = scores.masked_fill(mask == 0, -1e9) 
    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    attention_scores = torch.matmul(attention_weights, value)
    return attention_scores, attention_weights


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention: multiple attention channels."""
    
    def __init__(self, number_heads, embed_size, dropout_rate=0.1):
        """Take in model size and number of heads.
        
        Args:
            number_heads: number of heads.
            embed_size: embedding size.
            dropout_rate: dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        # We assume embedding is divisible by number of heads.
        #   e.g. 512/8 = 64, 8 heads, each head looks at 64d.
        assert embed_size % number_heads == 0
        self.attention_size = embed_size // number_heads 
        self.number_heads = number_heads
        self.linears = clones(nn.Linear(embed_size, embed_size), 4) # 4: see Vaswani/17.
        self.attention = None
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: shape = <batch-size, number-heads, seq-length, embed-size/number-heads>.
                   where attention-size = embed-size/number-heads.
            key: same shape as `query`.
            value: same shape as `query`.
            mask: <batch-size, 1, 1, seq-length>.
        Returns:
            Multi-head attended inputs, shape = <batch-size, seq-length, embed-size>.
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
                # <bc,1,mt> -> <bc,1,1,mt>
        number_batches = query.size(0)

        # 1) Do all the linear projections in batch from embed-size => h x attention-size
        # Operation 1. zip(l, (q,k,v)) -> (l,q), (l,k), (l,v)
        # Operation 2. l(x): <batch-size, seq-length, embed-size> 
        #                 -> <batch-size, seq-length, embed-size>
        # Operation 3. view: <batch-size, seq-length, embed-size> 
        #                 -> <batch-size, seq-length, number-heads, attention-size>
        # Operation 4. transpose: <batch-size,-1, number-heads, attention-size> 
        #                      -> <batch-size, number-heads, seq-length, attention-size>
        # NB: q, k, v are all of the shape <batch-size, number-heads, seq-length, attention-size>
        query, key, value = \
            [l(x).view(number_batches, -1, self.number_heads, self.attention_size).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
            
        # 2) Apply attention on all the projected vectors in batch.
        #   attention_scores: <batch-size, number-heads, seq-length,attention-size>
        #   attention_weights: <batch-size, number-heads, seq-length, seq-length>, 
        #     where the first seq-length is that of the attender.
        # NB: for the decoder-attends-encoder
        #     source_attention.attention shape: <batch-size, number-heads, decoder-seq-length, encoder-seq-length>.
        attention_scores, attention_weights = attention(query, key, value, mask=mask, 
                                                        dropout=self.dropout)
        self.attention = attention_weights 

        # 3) "Concat" using a view and apply a final linear. 
        # Operation 1. transpose: <batch-size, number-heads, seq-length, attention-size> 
        #                      -> <batch-size, seq-length, number-heads, attention-size>.
        # Operation 2. view: <batch-size, seq-length, number-heads*attention-size=embed-size>.
        attention_scores = attention_scores.transpose(1, 2).contiguous() \
                           .view(number_batches, -1, self.number_heads * self.attention_size)
        
        # 4) Apply the final linear layer.
        #  we have 4 linear layers in total, first 3 are for query, key and value,
        #  the last is here.
        #  Output shape = <batch-size, seq-length, embed-size>.
        return self.linears[-1](attention_scores)
    
    
class PositionwiseFeedForward(nn.Module):
    """Implements upsampling FFNN equation."""
    
    def __init__(self, embed_size, upsample_size, dropout_rate=0.1):
        """
        Args:
            embed_size: (main) hidden size.
            upsample_size: linear `blow-up` size, usually > embed_size.
            dropout_rate: dropout rate.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.linear_upsample = nn.Linear(embed_size, upsample_size)
        self.linear_downsample = nn.Linear(upsample_size, embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        Args:
            inputs: <batch-size, seq-length, embed-size>.
        Returns:
            Up-then-down sampled inputs of shape <batch-size, seq-length, embed-size>.
        """
        # Operation 1. linear: <batch-size, seq-length, embed-size> 
        #                   -> <batch-size, seq-length, upsample-size>
        # Operation 2. relu + dropout: shape retains.
        # Operation 3. linear: <batch-size, seq-length, upsample-size> 
        #                   -> <batch-size, seq-length, embed-size>
        return self.linear_downsample(self.dropout(F.relu(self.linear_upsample(inputs))))

    
class Embeddings(nn.Module):
    """Embedding lookup."""
    
    def __init__(self, embed_size, vocab_size, glove_init=None):
        """
        Args:
            embed_size: embedding size.
            vocab_size: vocab_size size.
            glove_init: numpy.ndarray, initializing embeddings.
        """
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        if glove_init is not None:
            assert glove_init.shape == (vocab_size, embed_size)
            self.embeddings.weight.data.copy_(torch.from_numpy(glove_init))
        self.embed_size = embed_size

    def forward(self, inputs):
        """
        Args:
            inputs: <batch-size, seq-length>.
        """
        # Lookup embeddings: <batch-size, seq-length> 
        #                 -> <batch-size, seq-length, embed-size>
        return self.embeddings(inputs) * math.sqrt(self.embed_size)
           
        
class PositionalEncoding(nn.Module):
    """Implement the Positional Encoding function."""
    
    def __init__(self, embed_size, dropout_rate, max_length=5000):
        """
        Args:
            embed_size: embedding size.
            dropout_rate: dropout rate.
            max_length: maximum of positions encodable.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_size, 2) *
                                  -(math.log(10000.0) / embed_size))
        positional_encoding[:, 0::2] = torch.sin(position * division_term)
        positional_encoding[:, 1::2] = torch.cos(position * division_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        # Adds a persistent buffer to the module.
        self.register_buffer("positional_encoding", positional_encoding)
        
    def forward(self, inputs):
        """
        Args:
            inputs: shape = <batch-size, seq-length, embed-size>
        """
        inputs = inputs + Variable(self.positional_encoding[:, :inputs.size(1)], 
                         requires_grad=False)
        return self.dropout(inputs)


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""
    
    # Label smoothing actually starts to penalize the model if it gets very 
    #   confident about a given choice.
    def __init__(self, size, padding_index, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_index = padding_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_distance = None
        
    def forward(self, prediction, target):
        assert prediction.size(1) == self.size
        true_distance = prediction.data.clone()
        true_distance.fill_(self.smoothing / (self.size - 2))
        true_distance.scatter_(1, target.long().data.unsqueeze(1), 
                               self.confidence)
        true_distance[:, self.padding_index] = 0
        mask = torch.nonzero(target.data == self.padding_index)
        if mask.dim() > 0:
            true_distance.index_fill_(0, mask.squeeze(), 0.0)
        self.true_distance = true_distance
        return self.criterion(prediction, Variable(true_distance, requires_grad=False))
    

class NoamOpt:
    """Optimizer wrapper that implements rate."""
    
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
    

class SimpleLossCompute:
    """A simple loss compute and train function."""
    
    def __init__(self, generator, criterion, noam_opt=None):
        self.generator = generator
        self.criterion = criterion
        self.noam_opt = noam_opt
        
    def __call__(self, outputs, target, norm):
        """
        Args:
            outputs: <batch-size, seq-length, embed-size>
            target: <batch-size, seq-length>
        Returns:
            Float loss.
        """
        # Apply final layer: <batch-size, seq-length, embed-size> 
        #                 -> <batch-size, seq-length, vocab-size>
        prediction = self.generator(outputs)
        # Shape changes:
        #   prediction: <batch-size, seq-length, vocab-size> 
        #            -> <batch-size*seq-length, vocab-size>
        #   target: <batch-size, seq-length> 
        #        -> <batch-size*seq-length,>
        # NB: criterion requires both to be of type torch.FloatTensor.
        prediction = prediction.contiguous().view(-1, prediction.size(-1))
        target = target.contiguous().view(-1)
        loss = self.criterion(prediction, target.float()) / norm.float()
        loss.backward()
        if self.noam_opt is not None:
            self.noam_opt.step()
            self.noam_opt.optimizer.zero_grad()
        return loss.float().item() * norm.float()