import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np


class PositionalEncoder:
    def __init__(self, input_dims, embedding_dims, encoder=True):
        self.input_dims = input_dims
        self.embedding = np.linspace(0, embedding_dims)
        self.embedding_dims = embedding_dims
        self.encoder = encoder

    def encode(self, tokenized_inputs):
        ## inputs np.array with dims (batch_size, input_dims, embedding_dims)
        ## returns np.array with dims (batch_size, input_dims, embedding_dims)
        encodings = []
        for i in range(tokenized_inputs_enc.shape[-2]):
            denominator = (i / 10000) ** (2*self.embedding/self.embedding_dims)
            if self.encoder:
                encodings.append(np.sin(i / denominator))
            else:
                encodings.append(np.cos(i / denominator))
        pos_enc = np.stack(encodings_encoder)
        return np.multiply(pos_enc, tokenized_inputs)
   

class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads, mask=False):
        super(MultiHeadedAttention, self).__init__()
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.norm_factor = np.sqrt(embedding_dims)
        self.qkv_dims = embedding_dims // n_heads
        self.n_heads = n_heads
    
        self.q_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)
        self.k_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)
        self.v_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)

        self.output_proj = nn.Linear(self.embedding_dims, self.embedding_dims, bias=False)
        self.mask = mask


    def forward(self, encoded_vectors, decoder_values=None):
        batch_size = encoded_vectors[0]
        new_dims = (batch_size, self.input_dims, self.n_heads, self.qkv_dims)
        
        queries = self.q_proj(encoded_vectors)
        keys = self.k_proj(encoded_vectors)
        
        if decoder_values == None: 
            values = self.v_proj(encoded_vectors)
            skip_val = values
        else: 
            values = decoder_values
            skip_val = values

        queries = queries.reshape(*new_dims)
        keys = keys.reshape(*new_dims)
        values = values.reshape(*new_dims)
        ## dims (batch_size, n_heads, input_dims, qkv_dims)

        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        ## dims (batch_size, n_heads, input_dims, qkv_dims)

        out = T.einsum('stuv, stvw -> stuw', queries, keys.transpose(-2, -1))
        mask = T.zeros_like(out, dtype=T.float32)
        if self.mask:
            mask = T.ones_like(out, dtype=T.float32) * float('-inf')
            mask = T.triu(mask, diagonal=1)

        out = (out + mask) / self.norm_factor
        out = F.softmax(out, dim=-1)
        ## dims (batch_size, n_heads, input_dims, input_dims)

        attention_values = T.einsum('stuu, stuv -> stuv', \
                                    out, values).permute(0, 2, 1, 3).contiguous()
        ## dims (batch_size, input_dims, n_heads, qkv_dims)

        attention_values = attention_values.flatten(start_dim=2)
        ## dims (batch_size, input_dims, embedding_dims)

        return self.output_proj(attention_values) + skip_val



class Encoder(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads):
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.n_heads = n_heads
        
        self.mha = MultiHeadedAttention(input_dims, embedding_dims, n_heads)
        self.norm_1 = nn.LayerNorm(emedding_dims)

        self.mlp = nn.Sequential(
                nn.Linear(embedding_dims, 4 * embedding_dims),
                nn.LeakyReLU(),
                nn.Linear(4 * embedding_dims, embedding_dims)
                )
        self.norm_2 = nn.LayerNorm(embedding_dims)


    def forward(self, inputs):
        mha_output = self.norm_1(self.mha.forward(inputs))
        mlp_output = self.norm_2(self.mlp(mha_output) + mha_output)
        return mlp_output

        

class Decoder(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads):
        super(Decoder, self).__init__()
        self.input_dims = input_dims
        self.embedding_dims = embedding_dims
        self.n_heads = n_heads

        self.mha_masked = MultiHeadedAttention(input_dims, embedding_dims, \
                                                n_heads, mask=True)
        self.norm_1 = nn.LayerNorm(embedding_dims)
        
        self.mha_joined = MultiHeadedAttention(input_dims, embedding_dims, n_heads)
        self.norm_2 = nn.LayerNorm(embedding_dims)

        self.mlp = nn.Sequential(
                nn.Linear(embedding_dims, 4 * embedding_dims),
                nn.LeakyReLU(),
                nn.Linear(4 * embedding_dims, embedding_dims)
                )
        self.norm_3 = nn.LayerNorm(embedding_dims)


    def forward(self, encoder_ouputs, decoder_vectors):
        values = self.norm_1(self.mha_masked.forward(decoder_vectors))
        
        out_joined = self.norm_2(self.mha_joined.forward(encoder_outputs, decoder_vectors)) 
        return self.norm_3(self.mlp(out_joined))


class Transformer(nn.Module):
    def __init__(self, input_dims, embedding_dims, n_heads, output_dims, lr, n_blocks=4, decoder=False):
        super(Transformer, self).__init__()
        self.decoder = decoder
        self.pos_encoder_enc = PositionalEncoder(input_dims, embedding_dims, encoder=True)
        self.encoder = Encoder(input_dims, embedding_dims, n_heads)
        if decoder:
            self.pos_encoder_dec = PositionalEncoder(input_dims, embedding_dims, encoder=False)
            self.decoder = Decoder(input_dims, embedding_dims, n_heads)
        self.final_proj = nn.Linear(embedding_dims, output_dims)
        self.n_blocks = n_blocks

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        encoder_outputs = self.pos_encoder_enc.forward(inputs)
        for _ in range(self.n_blocks):
            encoder_outputs = self.encoder.forward(encoder_outputs)

        if not self.decoder:
            probs = F.softmax(self.final_proj(encoder_outputs), dim=-1)
            return probs
        
        decoder_outputs = self.pos_encoder_enc.forward(inputs)
        for _ in range(self.n_blocks):
            decoder_outputs = self.decoder(encoder_outputs, decoder_outputs)
        probs = F.softmax(self.final_proj(decoder_outputs), dim=-1)
        return probs




