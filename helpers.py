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
Creates an encoding for each word within the input text representing the position
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

        # Since each word is represented as a vector, we need to add positional encoding information to the entire vector
        positional_encoding = torch.zeros(seq_len, d_model)

        '''
            ToDo: Better understand the reasoning of the below
            - Formula: Sin(pos / 10,000 ^ (2i / d_model))  => Use Cos for odd positions
            - We apply a different encoding for each item of the word embedding vector based
                based on if the position is odd or even (not the position within the sentence!)
            - Use a log function instead of original sin/cos for numerical stability
                - loga(b) = c is equivalent to a^c = b
                - 1 / 10,000 ^ (2i / d_model) => 10,000 ^ -(2i / d_model) => 
        '''     

        # Create an ascending vector of shape (seq_len, 1) representing position of word in sentence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # unsqueeze just adds another dimension
        # Divisor has step size 2 because we'll only apply it to either even or odd indices
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model) ) #
        # Apply to even and odd positions
        positional_encoding[:, 0::2] = torch.sin(position * divisor)
        positional_encoding[:, 1::2] = torch.sin(position * divisor) 
        # Add batch size since we will process multiple sentences as once (1, seq_len, d_model)
        positional_encoding = positional_encoding.unsqueeze(1)

        # Saves the tensor in buffer of module
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('positional_encoding', positional_encoding, persistent=False)


    def forward(self, x):
        # Add positional encoding for JUST this sentence
        # Positional encoding should be static, so turn off gradients
        x =  x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)





