import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ChessNetworkConvolutional(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ChessNetworkConvolutional, self).__init__()
        input_dims = input_dims[-2] * input_dims[-1]
        self.actor_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dims, 1),
            nn.Tanh()
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def block(self, in_filters=128):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_filters, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128)
        )
        return block

    def downsample(self, state, out_chan=128, stride=1):
        in_chan = state.shape[1]
        if in_chan == out_chan:
            return state
        downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_chan)
        )
        downsample = downsample.to(self.device)
        return downsample(state)

    def connect_residual(self, state, block_output, out_chan=128, stride=1):
        residual_connection = self.downsample(state, out_chan, stride)
        return F.relu(block_output + residual_connection)


## Baseline vison-esque transformer network

class PositionalEncoding(nn.Module):
    def __init__(self, input_dims, encoding_dims=768, first_block=False):
        super(PositionalEncoding, self).__init__()
        self.first_block = first_block

        self.encodings = nn.Conv2d(
            in_channels=6,
            out_channels=encoding_dims,
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=0
            )
        patch_dims = (input_dims[-2] // 2) * (input_dims[-1] // 2)
        if first_block:
            self.cls_token = nn.Parameter(T.zeros(1, 1, encoding_dims))
            self.pos_embed = nn.Parameter(T.zeros(1, 1 + patch_dims, \
                encoding_dims))
        self.dropout = nn.Dropout(p=.2)

    def forward(self, inputs):
        if not self.first_block:
            return inputs
        ## input_dims (N, in_channels, height, width)
        encodings = self.encodings(inputs)
        encoded_vectors = T.transpose(encodings.flatten(start_dim=2), 1, 2)
        ## encoded dims (N, n_patches, encoding_dims)
        cls_token = self.cls_token.expand(inputs.shape[0], -1, -1)
        encoded_vectors = T.cat((cls_token, encoded_vectors), dim=1)
        encoded_vectors += self.pos_embed
        ## encoded dims (N, n_patches, encoding_dims)
        encoded_vectors = self.dropout(encoded_vectors)
        return encoded_vectors

class MultiHeadedAttention(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, first_block=False):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.head_dims = encoding_dims // n_heads
        self.norm_factor = np.sqrt(self.head_dims)
        self.encoder = PositionalEncoding(input_dims, encoding_dims, first_block)

        self.queries = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.keys = nn.Linear(encoding_dims, encoding_dims, bias=False)
        self.values = nn.Linear(encoding_dims, encoding_dims, bias=False)

        self.dropout = nn.Dropout(p=.2)

        self.fc = nn.Linear(encoding_dims, encoding_dims, bias=False)

    def forward(self, inputs):
        encoded_vectors = self.encoder.forward(inputs)
        ## encoded dims (N, n_patches, encoding_dims)

        batch_size, n_patches, _ = encoded_vectors.shape

        queries = self.queries(encoded_vectors)
        keys = self.keys(encoded_vectors)
        values = self.values(encoded_vectors)
        ## qkv dims (N, n_patches, encoding_dims)

        queries = queries.reshape(batch_size, n_patches, self.n_heads, self.head_dims)
        keys = keys.reshape(batch_size, n_patches, self.n_heads, self.head_dims)
        values = values.reshape(batch_size, n_patches, self.n_heads, self.head_dims)
        ## qkv dims (N, n_patches, n_heads, head_dims)

        queries = queries.permute(0, 2, 1, 3).contiguous()
        keys = keys.permute(0, 2, 1, 3).contiguous()
        values = values.permute(0, 2, 1, 3).contiguous()
        ## qkv dims (N, n_heads, n_patches, head_dims)

        out = T.einsum('stuv, stvw -> stuw', queries, keys.transpose(-2, -1)) / self.norm_factor
        out = F.softmax(out, dim=-1)
        ## out dims (N, n_heads, n_patches, n_patches)
        attention_values = T.einsum('stuu, stuv -> stuv', out, values).permute(0, 2, 1, 3).contiguous()
        ## att val dims (N, n_patches, n_heads, head_dims)
        attention_values = attention_values.flatten(start_dim=2)
        ## att val dims (N, n_patches, encoding_dims)
        attention_values = self.dropout(self.fc(attention_values))
        return attention_values, encoded_vectors


class TransformerEncoder(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, \
        first_block=False):
        super(TransformerEncoder, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(
            input_dims,
            encoding_dims,
            n_heads,
            first_block
            )

        self.norm_1 = nn.LayerNorm(encoding_dims)

        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dims, encoding_dims*4),
            nn.LayerNorm(encoding_dims*4),
            nn.Dropout(p=.2),
            nn.GELU(),
            nn.Linear(encoding_dims*4, encoding_dims),
            nn.Dropout(p=.2)
        )

        self.norm_2 = nn.LayerNorm(encoding_dims)

    def forward(self, inputs):
        mha_output, skip_val = self.multi_headed_attention.forward(inputs)
        add_and_norm = self.norm_1(skip_val + mha_output)
        ff_output = self.feed_forward(add_and_norm)
        output = self.norm_2(add_and_norm + ff_output)
        ## output dims (N, n_patches, encoding_dim)
        return output

class TransformerNetwork(nn.Module):
    def __init__(self, input_dims, encoding_dims, n_heads, \
        n_encoder_blocks):
        super(TransformerNetwork, self).__init__()
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(input_dims, encoding_dims, n_heads, \
                first_block=True)] + \
            [TransformerEncoder(input_dims, encoding_dims, n_heads) \
                for _ in range(n_encoder_blocks-1)]
        )
        self.network = nn.Sequential(
            *self.encoder_blocks,
            nn.LayerNorm(encoding_dims)
        )


class ChessNetworkTransformer(nn.Module):
    def __init__(self, input_dims, n_actions, encoding_dims=768, n_heads=12, \
        n_encoder_blocks=4):
        super(ChessNetworkTransformer, self).__init__()
        transformer = TransformerNetwork(input_dims, encoding_dims, n_heads, \
            n_encoder_blocks)
        self.network = transformer.network

        self.actor_head = nn.Sequential(
            nn.Linear(encoding_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_head = nn.Sequential(
            nn.Linear(encoding_dims, 1),
            nn.Tanh()
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        out = self.network(input)[:, 0, :]
        ## out dims {just cls token} (N, encoding_dims)
        probs = self.actor_head.forward(out)
        value = self.critic_head.forward(out)
        return probs, value