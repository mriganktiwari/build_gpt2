import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # [query | key | value]
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1,1,config.block_size, config.block_size),
                            persistent=False)
        
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.c_attn(x) # b,t,3c
        q,k,v = qkv.split(config.n_embd, dim=-1) # b,t,c for each q,k,v
        q = q.view(B, T, self.n_head, self.head_size).transpose(1,2) # b,nh,t,hs
        k = k.view(B, T, self.n_head, self.head_size).transpose(1,2) # b,nh,t,hs
        v = v.view(B, T, self.n_head, self.head_size).transpose(1,2) # b,nh,t,hs
    
        affinity_scores = q @ k.transpose(-2,-1) * (1/math.sqrt(k.size(-1))) # b,nh,t,hs @ b,nh,hs,t -> b,nh,t,t
        affinity_scores = affinity_scores.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf')) # b,nh,t,t
        affinity_scores = F.softmax(affinity_scores, dim=-1)

        y = affinity_scores @ v # b,nh,t,hs, but need b,t,c
        y = y.transpose(1,2).contiguous().view(B,T,C) # b,t,c
        y = self.c_proj(y) # b,t,c -> b,t,c
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # fully connected layer
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)  # projection layer

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = self.attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h.0, h.1 ... will come from this list
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        config_args = {
            'gpt2': dict(n_embd=768, n_layer=12, n_head=12),
            'gpt2-medium': dict(n_embd=1024, n_layer=24, n_head=16),
            'gpt2-large': dict(n_embd=1280, n_layer=36, n_head=20),
            'gpt2-xl': dict(n_embd=1600, n_layer=48, n_head=25),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)

        config = GPTConfig(**config_args)
        model = GPT2(config=config)

        sd = model.state_dict()
        sd_hf = model_hf.state_dict()

        sd_keys = sd.keys()
        sd_keys_hf = sd_hf.keys()

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f'keys mismatched: {len(sd_keys)} != {len(sd_keys_hf)}'
        for k in sd_keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

# config = GPTConfig()
# model = GPT2(config=config)
# for k,v in model.state_dict().items():
#     print(f'{k} --> {v.shape}')

model = GPT2.from_pretrained('gpt2')
print(f'weights loaded form HF !')