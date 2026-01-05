
import torch as tr
from torch import nn
from torch.utils.data import Dataset
import numpy as np
'''
    doing the above encoding in PyTorch
'''

class GPTDataset(Dataset): ## inherit the Dataset class from pytorch

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            ''' stride shifts only the input, the target is always shifted by 1'''
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(tr.tensor(input_chunk))
            self.target_ids.append(tr.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
    
#-----------------------------------------------------------------------------

 ## we implement a simple self attention class
class SelfAttention(tr.nn.Module):
    def __init__(self,d_in,d_out):
        super(SelfAttention,self).__init__()
        self.W_query = nn.Linear(d_in,d_out)
        self.W_key = nn.Linear(d_in,d_out)
        self.W_value = nn.Linear(d_in,d_out)
    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attention_scores = tr.nn.functional.softmax(queries @ keys.mT,dim=-1)
        context_vectors = tr.matmul(attention_scores,values)
        return context_vectors

## 
class CausalSelfAttention(SelfAttention):
    def __init__(self, d_in, d_out, context_length, dropout_prob):
        super(CausalSelfAttention,self).__init__(d_in,d_out) ## use the init of parent class
        self.dropout = nn.Dropout(p=dropout_prob)
        self.register_buffer("mask", tr.triu(tr.ones(context_length, context_length), diagonal=1))
    def forward(self, x):   ## overloading the forward method
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x) ## dim of values = batches x num_tokens x output dimen
        num_tokens = x.shape[1] ## shape of x = batches x num_tokens x embedding dim
        self.mask = tr.tril(tr.ones((num_tokens,num_tokens))).to(dtype=tr.bool) ## upper triangular matrix
        attention_scores = tr.matmul(queries,keys.mT) ## unnormalized scores
        attention_scores = attention_scores.masked_fill(~self.mask, -tr.inf) ## mask future tokens
        attention_scores = tr.nn.functional.softmax(attention_scores,dim=-1) ## dimen = batches x num_tokens X num_tokens
        # print(attention_scores)
        # attention_scores = self.dropout(attention_scores)
        context_vectors = tr.matmul(attention_scores,values) ## dimen = batches x num_tokens x output dimen
        return context_vectors
    

class MultiHeadAttention(CausalSelfAttention):
    def __init__(self, d_inp, d_out, context_length, dropout_prob, num_heads, qkv_bias):
        super(MultiHeadAttention,self).__init__(d_inp, d_out, context_length, dropout_prob)
        self.W_query = nn.Linear(d_inp,d_out,bias=qkv_bias)
        self.W_key = nn.Linear(d_inp,d_out,bias=qkv_bias)
        self.W_value = nn.Linear(d_inp,d_out,bias=qkv_bias)
        self.d_inp = d_inp
        self.d_out = d_out ## concatenated dimension from all attention heads
        self.dim_head = d_out//num_heads
        self.num_heads = num_heads
        self.out_proj = nn.Linear(d_out,d_out)
        self.register_buffer("mask", tr.triu(tr.ones(context_length, context_length), diagonal=1))

    def forward(self,x):
        ## W also have d_out, the concatenated outputs
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        batches, seq_len, _ = x.shape
        ## we split the output dim of quesries in num_head x dim_head 
        queries = queries.view(batches,seq_len,self.num_heads,self.dim_head)
        keys = keys.view(batches,seq_len,self.num_heads,self.dim_head)
        values = values.view(batches,seq_len,self.num_heads,self.dim_head)

        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # print(queries.shape,keys.shape,values.shape)
        attention_scores = tr.matmul(queries,keys.transpose(2,3)) 
        # self.mask = tr.tril(tr.ones((batches, self.num_heads,seq_len,seq_len))).to(dtype=tr.bool) ## upper triangular matrix
        # print(mask.shape,attention_scores.shape)
        mask_bool = self.mask.bool()[:seq_len, :seq_len]

        attention_scores = attention_scores.masked_fill(mask_bool, -tr.inf) ## mask future tokens
        attention_scores = tr.nn.functional.softmax(attention_scores/(keys.shape[-1]**0.5),dim=-1) ## scaled, masked, normalized attention
        
        context_vectors = tr.matmul(attention_scores,values).transpose(1,2) ## tranpose again to get #head x dim_heads as last 2 dimen
        context_vectors = context_vectors.contiguous().view(batches,seq_len, self.d_out)  ## dimen = batches x num_tokens x output dimen
        context_vectors = self.out_proj(context_vectors)
        return context_vectors

        
#---------------------------------------------------------------------------------

class FFN(nn.Module):
    def __init__(self,GPT_config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(GPT_config["emb_dim"],4*GPT_config["emb_dim"]),
                                    nn.GELU(),
                                    nn.Linear(4*GPT_config["emb_dim"],GPT_config["emb_dim"]))
    def forward(self,x):
        return self.layers(x)

class LayerNormalization(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(tr.ones(emb_dim))
        self.shift = nn.Parameter(tr.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / tr.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class Transformer(nn.Module):
    def __init__(self,gpt_config):
        super().__init__()
        
        # eps = 1e-6
        self.Layer_norm1 = LayerNormalization(gpt_config["emb_dim"])
        self.Layer_norm2 = LayerNormalization(gpt_config["emb_dim"])
        self.Multihead_attn = MultiHeadAttention(gpt_config["emb_dim"],gpt_config["emb_dim"],gpt_config["context_length"],gpt_config["drop_rate"],gpt_config["n_heads"],gpt_config["qkv_bias"])
        self.FFN = FFN(gpt_config)
        self.Dropout = nn.Dropout(gpt_config["drop_rate"])
        
    def forward(self,x):

        attn_output = self.Layer_norm1(x)
        attn_output = self.Multihead_attn(attn_output)
        attn_output = self.Dropout(attn_output)
        attn_output = x + attn_output

        FFN_output = self.Layer_norm2(attn_output)
        FFN_output = self.FFN(FFN_output) 
        FFN_output = self.Dropout(FFN_output)

        return attn_output + FFN_output


class GPTModule(nn.Module):
    def __init__(self, gpt_config):
        super().__init__()
        self.vocab_size = gpt_config["vocab_size"]
        self.context_length = gpt_config["context_length"] 
        self.emb_dim = gpt_config["emb_dim"] 
        self.n_heads  = gpt_config["n_heads"] 
        self.n_layers = gpt_config["n_layers"]
        self.drop_rate = gpt_config["drop_rate"]
        self.qkv_bias = gpt_config["qkv_bias"]
        
        self.Token_Embedding = nn.Embedding(self.vocab_size,self.emb_dim)
        self.Position_Embedding = nn.Embedding(self.context_length,self.emb_dim)
        self.Dropout_bef_Transform = nn.Dropout(self.drop_rate)
        self.Transformer_block = nn.Sequential(*[Transformer(gpt_config) for _ in range(self.n_layers)]) 
        self.Final_Layer_Norm = LayerNormalization(gpt_config["emb_dim"])
        self.Final_output = nn.Linear(self.emb_dim,self.vocab_size,bias=False)
        
    def forward(self,x):
        seq_len = x.shape[1]
        x = self.Token_Embedding(x) + self.Position_Embedding(tr.arange(seq_len, device=x.device))   
        x = self.Dropout_bef_Transform(x)
        x = self.Transformer_block(x)
        x = self.Final_Layer_Norm(x)

        return self.Final_output(x)
