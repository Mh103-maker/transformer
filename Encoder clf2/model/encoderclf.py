import math
from types import new_class
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy

from .embedding import TransformerEmbedding

# clone layer 레이어 수 만큼 복사
def clones(module, N):
    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

###########

class Encoderclf(nn.Module):

    def __init__(self, src_vocab, n_class, N , d_model,h, d_ff, dropout):
        super().__init__() 
        
        self.c=copy.deepcopy
        self.attn=MultiHeadedAttention(h=h, d_model=d_model)
        self.ff= PositionwiseFeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.encoder=Encoder(EncoderLayer(d_model, self.c(self.attn),self.c(self.ff),dropout),N) # 인코더 전체
        self.src_embed=TransformerEmbedding(d_model=d_model, vocab=src_vocab, dropout=dropout) #embedding class
        
        self.generator=Generator(d_model=d_model, n_class=n_class)       

    def forward(self, src):
        x=self.encoder(self.src_embed(src))
        return x

    def encode(self, src):
        return self.encoder(self.src_embed(src))

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, n_class ):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model,n_class)

    def forward(self, x):
        #print("transformer 결과",x.size()) #batch, seq_len, d_model
        x= x.mean(dim=1) # batch ,d_model
        #print("평균결과",x.size())
        x=self.proj(x) # batch, n_class
        #print("proj결과",x.size())
        x=log_softmax(x,dim=-1)
        #print("softmax결과",x.size())
        return x # batch, n_class

############
class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder,self).__init__()
        self.layers=clones(layer,N) #encoder layer N개만큼
        self.norm=LayerNorm(layer.size) #layer.size=d_model

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        return self.norm(x)


########
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    "feature 차원 정규화"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  #d_model
        std = x.std(-1, keepdim=True) 
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    dropout -> residual connection
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn #Multiheadattention 예정
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

###########
def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    p_attn = scores.softmax(dim=-1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
        
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
            
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        #파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추
        # transpose -> e두개의 차원 교환
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

#############
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation.-Nonlinearity"

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))    #max(0,xW_1+b_1)W_2 +b_2