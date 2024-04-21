import torch
import torch.nn as nn
import math
from torch import Tensor


'''
Embedding for input text.
Each word has an ID in the language, which maps to a vector of size d_model.
The embedding layer is basically a lookup table that the model will learn from,
    meaning we can't change the embedding after training
'''
class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.sqrt_d_model = math.sqrt(self.d_model)
        self.embedding = nn.Embedding(vocab_size, d_model) # Lookup table

    def forward(self, x) -> Tensor:
        return self.embedding(x) * self.sqrt_d_model
    

'''
Gives position of the token within the input text.
d_model here is the same as the length of the embedding vector in InputEmbeddings
'''
class PositionalEncoding(nn.Module):

    '''
        d_model = Length of the embedding vector from InputEmbeddings that we add positions to (length of a word representation)
        seq_len = Length of the entire sequence (number of words)
        p_dropout = The probability to apply dropout
    '''
    def __init__(self, d_model: int, seq_len: int, p_dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p_dropout)

        # Since each word is represented as a vector, we need to add positional
        # encoding information to the entire vector
        positional_encoding = torch.zero(seq_len, d_model)

        '''
            We apply a different encoding based on if the position is odd or even.
            Use a log function instead of original sin/cos for numerical stability
        '''
