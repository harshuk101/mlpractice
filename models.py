import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


# https://arxiv.org/pdf/1706.03762v7.pdf
# https://www.youtube.com/watch?v=ISNdQcPhsts
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class Transformer(nn.Module):
    def __init__(self,  num_encoder_layers:int = 6):
        super().__init__()

        self.FFLayers = None


class MultiHeadAttention(nn.Module):

    #Input: T vectors of size M => T x M where each vector is a word
    def __init__(self, len_embedding = 8, num_heads = 4):
        super().__init__()
        self.len_embedding, self.num_heads = len_embedding, num_heads
        self.q = nn.Linear(len_embedding, len_embedding, biase=False)
        self.k = nn.Linear(len_embedding, len_embedding, biase=False)
        self.v = nn.Linear(len_embedding, len_embedding, biase=False)

    def forward(self, queries, key, values):
        ## ToDo
        return None


    def attention(self, input):
        return None