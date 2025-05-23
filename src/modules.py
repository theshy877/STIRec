# -*- coding:utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from param import args


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
    (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        """Construct an RMSNorm module."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        x = x / rms
        return self.weight * x

class LongSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({args.hidden_size}) is not a multiple of the number of attention "
                f"heads ({args.num_attention_heads})"
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.L = args.avg_window_size  # Moving average window size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.RMSNorm = RMSNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_multihead(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: Tensor, attention_mask: Tensor):
        batch_size, seq_len, hidden_size = input_tensor.size()
        seq_avg = input_tensor.clone()
        mask_avg = (attention_mask[:, 0, 0, :] > -1e4).float().unsqueeze(-1)  # [batch_size, seq_len, 1]

        for l in range(1, self.L):
            padding = torch.zeros(batch_size, l, hidden_size, device=input_tensor.device)
            mask_padding = torch.zeros(batch_size, l, 1, device=input_tensor.device)
            
            seq_avg = seq_avg + torch.cat([padding, input_tensor[:, :-l, :]], dim=1)
            mask_avg = mask_avg + torch.cat([mask_padding, mask_avg[:, :-l, :]], dim=1)
        
        seq_avg = seq_avg / (mask_avg + 1e-12)
        seq_avg_norm = self.LayerNorm(seq_avg)

        mixed_query_layer = self.query(seq_avg_norm)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_multihead(mixed_query_layer)
        key_layer = self.transpose_multihead(mixed_key_layer)
        value_layer = self.transpose_multihead(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + seq_avg_norm)

        return hidden_states
class SelfAttention(nn.Module):
    """Self-attention module for SASRec."""

    def __init__(self):
        super().__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({args.hidden_size}) is not a multiple of the number of attention "
                f"heads ({args.num_attention_heads})"
            )
        self.num_attention_heads: int = args.num_attention_heads
        self.attention_head_size: int = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size: int = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.RMSNorm = RMSNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_multihead(self, x):
        """Transpose to multi head attention shape.

        e.g., [256, 50, 64] -> [256, 50, 2, 32] -> [256, 2, 50, 32]
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor: Tensor, attention_mask: Tensor):
        """Self-attention:

        1. multi head attention
            - input + W matrix -> Q K V
            - Q K V -> Multihead
            - Q^T * K / sqrt(d_k)
            - mask attention score
            - softmax
            - dropout
        * No: 2. Add & Norm
            - weight sum
            - Layer Norm
        3. FFN
            - linear
            - dropout
        4. Add & Norm
            - residual connection
            - Layer Norm
        5. return

        Args:
            input_tensor (Tensor): [256, 50, 64] -> [batch_size, seq_len, hidden_size]
            attention_mask (Tensor): [256, 1, 50, 50] -> [batch_size, 1, seq_len, seq_len]

        Returns:
            Tensor: shape: [batch_size, seq_len, hidden_size]
        """
        # * shape: [batch_size, seq_len, hidden_size]
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # * shape: [batch_size, heads, seq_len, head_size]
        query_layer = self.transpose_multihead(mixed_query_layer)
        key_layer = self.transpose_multihead(mixed_key_layer)
        value_layer = self.transpose_multihead(mixed_value_layer)

        # * Take the dot product between "query" and "key" to get the raw attention scores.
        # * formula: Q^T * K / sqrt(d_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # * Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # * [batch_size heads seq_len seq_len] scores
        # * [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # * Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # * This is actually dropping out entire tokens to attend to, which might
        # * seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)

        # * formula: dropout(softmax(Q^T * K)) * V
        context_layer = torch.matmul(attention_probs, value_layer)

        # * shape: [batch_size, seq_len, heads, head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # * shape: [batch_size, seq_len, hidden_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # * Linear + dropout + residual + layer norm
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.RMSNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        """FFN: Feed forward network.
        1. linear
        2. activation
        3. linear
        4. dropout
        5. residual connection
        6. layer norm
        7. return

        Args:
            input_tensor (Tensor): [256, 50, 64] -> [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: shape: [batch_size, seq_len, hidden_size]
        """

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    """Layer block in the transformer model."""

    def __init__(self):
        super().__init__()
        self.attention = SelfAttention()
        self.intermediate_1 = Intermediate()
        self.intermediate_2 = Intermediate()
        self.long_attention = LongSelfAttention()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        long_attention_output = self.long_attention(hidden_states, attention_mask)
        return args.alpha * self.intermediate_1(attention_output) + (1 - args.alpha) * self.intermediate_2(long_attention_output)


class Encoder(nn.Module):
    """Encoder: a stack of N layers."""

    def __init__(self):
        super().__init__()
        layer = Layer()
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states: Tensor, attention_mask: Tensor, output_all_encoded_layers=True) -> list[Tensor]:
        """

        Args:
            hidden_states (Tensor): input. [256, 50, 64] -> [batch_size, seq_len, hidden_size]
            attention_mask (Tensor): [256, 1, 50, 50] -> [batch_size, 1, seq_len, seq_len]
            output_all_encoded_layers (bool, optional): if True, return output of all layers. Else, return the last layer only. Defaults to True.

        Returns:
            list[Tensor]: list of hidden states of all layers or the last layer. SHAPE: [Layer_num, batch_size, seq_len, hidden_size] or [1, batch_size, seq_len, hidden_size
        """

        all_encoder_layers: list[Tensor] = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LigthGCNLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, adj, embeds):
        return torch.sparse.mm(adj, embeds)


class NGCFLayer(nn.Module):

    def __init__(self, in_features=args.hidden_size, out_features=args.hidden_size):
        super().__init__()
        # self.dropout = nn.Dropout(args.gcn_dropout_prob)
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, adj, embeds):
        embeds = self.linear(embeds)
        # embeds = self.dropout(embeds)
        return F.leaky_relu(torch.sparse.mm(adj, embeds), negative_slope=0.2)
