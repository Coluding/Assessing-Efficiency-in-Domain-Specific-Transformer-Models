import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional
import xformers.components
from xformers.components.attention import (
    maybe_sparsify,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern
)
from xformers.components.multi_head_dispatch import MultiHeadDispatch


from src.attention import utils

class LocalAttentionDilation(xformers.components.attention.LocalAttention):
    """
    Extends the LocalAttention class to include dilation and global token functionality.

    This class allows for the application of local attention with an optional dilation rate,
    which can be useful in various sequence modeling tasks where context at varied scales is beneficial.
    Additionally global token can be set.

    Attributes:
        dilation_rate (int or None): Specifies the dilation rate for the attention mask.
                                     If None, standard local attention is applied.
    """

    def __init__(self, dropout: float = 0.0, causal: bool = False, window_size: int = 5, force_sparsity: bool = False, *args, **kwargs):
        """
       Initializer

        :param dropout:  Dropout probability.
        :param causal:  If True, ensures that the attention is causal.
        :param window_size: The size of the attention window.
        :param force_sparsity: If True, forces the attention mask to be sparse.
        :param args:  Variable length argument list.
        :param kwargs: Arbitrary keyword arguments.
        """
        super().__init__(dropout, causal, window_size, force_sparsity, *args, **kwargs)
        self.dilation_rate: Optional[int] = None
        self.global_indices: Optional[int] = None

    def _get_local_mask(self, shape: torch.Size) -> torch.Tensor:
        """
        Generates a local attention mask, potentially with dilation. Overwrites the current implementation in the
        xformer library.

        :param shape: The shape of the mask to be generated.
        :return: The generated attention mask.
        """

        if not isinstance(shape, torch.Size):
            raise ValueError("Expected 'shape' to be of type torch.Size")

        window_size = self.window_size * 2 + 1 if self.causal else self.window_size

        if self.dilation_rate is None or self.dilation_rate == 1:
            mask = local_1d_pattern(shape[1], window_size)
        else:
            if not isinstance(self.dilation_rate, int) or self.dilation_rate < 1:
                raise ValueError("Dilation rate must be a positive integer")
            mask = utils.local_1d_pattern_dilated(shape[1], window_size, self.dilation_rate)

        if self.global_indices is not None:
            mask = mask | utils.symmetric_global_token_mask(self.global_indices, shape[1])

        if self.causal:
            mask &= causal_1d_pattern(shape[1])

        mask = sparsify(mask) if self.force_sparsity else maybe_sparsify(mask)

        return mask

    def set_dilation(self, dilation_rate: int):
        """
        Sets the dilation rate for the attention mask.

        :param dilation_rate: The dilation rate to be applied.
        :return:
        """
        if not isinstance(dilation_rate, int) or dilation_rate < 1:
            raise ValueError("Dilation rate must be a positive integer")
        self.dilation_rate = dilation_rate

    def set_global_indices(self, global_indices: torch.Tensor):
        """Sets the global indices for the attention mask.

        :param global_indices: The indices.
        :return:
        """
        if not isinstance(global_indices, torch.Tensor):
            raise ValueError("Dilation rate must be a Tensor")
        self.global_indices = global_indices


class MultiHeadDilatedLocalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window_size: int,
                 global_tokens: Optional[torch.Tensor] = None,
                 dilation_rate: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.global_indices = global_tokens
        self.dilation_rate = dilation_rate
        
        local_attention = LocalAttentionDilation(window_size=window_size)

        if dilation_rate is not None:
            local_attention.set_dilation(dilation_rate)

        if global_tokens is not None:
            local_attention.set_global_indices(global_tokens)

        self.multihead = MultiHeadDispatch(attention=local_attention, dim_model=d_model//2, num_heads=num_heads)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        return self.multihead(q,k,v)




################################# Deprecated ############################################
class AttentionBlock(nn.Module):
    pass

class DilatedSlidingWindowAttention(nn.Module):
    def __init__(self, d_model: int, dilation_rate: int, window_size: int, global_token_indices: torch.Tensor,
                 num_heads: int, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.dilation_rate = dilation_rate
        self.window_size = window_size
        self.global_token_indices = global_token_indices
        self.device = device
        self.d_model = d_model

        self.mh_att_global = nn.MultiheadAttention(d_model, num_heads)
        self.mh_att_local = MultiHeadLocalAttention(d_model, num_heads)

        # will be set by the remove_global_attention_token_create_global_attention_tensor function of utils
        self.local_token_indices = torch.tensor(list(set(list(range(0, 2000))).difference(set(global_token_indices.tolist()))))

    # TODO: when k=q=v then only one operation would be needed and the sliced ones can be copied
    def _setup_global_local_attention(self, key: torch.Tensor, query: torch.Tensor, value: torch.Tensor):
        """
        Sets query = key

        :param key:
        :param query:
        :param value:
        :return:


        # also sets loca_token_indices attribute
           remove_global_attention_token_create_global_attention_tensor(key,
                                                                                                      self.global_token_indices,
                                                                                                      self)

        global_values, sliding_window_values = remove_global_attention_token_create_global_attention_tensor(value,
                                                                                                      self.global_token_indices)
        """
        global_keys = torch.randn(64,4,self.d_model).to("cuda")
        sliding_window_key = torch.randn(64, 1996, 1, self.d_model).to("cuda")
        global_queries = torch.randn(64, 4, self.d_model).to("cuda")
        sliding_window_queries = torch.randn(64, 1996, 1, self.d_model).to("cuda")
        """
        sliding_window_value = slice_tensor_in_windows(sliding_window_values, self.window_size, self.dilation_rate)
        sliding_window_query = slice_tensor_in_windows(sliding_window_queries, self.window_size, self.dilation_rate)
        """

        global_values = torch.randn(64, 4, self.d_model).to("cuda")
        sliding_window_value = torch.randn(64, 1996, 1, self.d_model).to("cuda")
       # sliding_window_key = rearrange(sliding_window_key, "b n d -> b n 1 d")

        return (global_keys, sliding_window_key), (global_queries, sliding_window_queries), (global_values, sliding_window_value)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor):

        final_attention_tensor = torch.zeros(*query.shape).to(self.device)
        queries, keys, values = self._setup_global_local_attention(key, query, value)
        local_attention = self.mh_att_local(queries[1], keys[1], values[1])
        global_attention = self.mh_att_global(queries[0], keys[0], values[0])

        final_attention_tensor[:, self.local_token_indices, :] = local_attention
        final_attention_tensor[:, self.global_token_indices, :] = global_attention[0]
        return final_attention_tensor


class BatchedLocalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
         Query of shape (batch_size x sequence_length x window_size x d_model).
        Value is of shape (batch_size x sequence_length x window_size x d_model). We peform the attention computation
        over 4 dimensions then which reduces the overall complexity to O(nw) where w is the window size.

        :param query: Query tensor
        :param key: Key tensor
        :param value: Value tensor
        :return: Attention output of shape sequence_length x d_model
        """
        d_model = query.shape[-1]

        # Equation (9) in the thesis
        S_prior: torch.Tensor = query @ key.transpose(-2, -1)
        S: torch.Tensor = self.sm(torch.div(S_prior, np.sqrt(d_model)))

        # Equation (10) in the thesis
        A = S @ value

        return A.squeeze()


class MultiHeadLocalAttention(nn.Module):
    """
    Custom Multi-Head Local Attention Module.
    This module is designed to perform local attention in a multi-head setup,
    processing the input with multiple attention heads in parallel.

    Attributes:
        num_heads (int): Number of attention heads.
        d_model (int): The dimension of the input embeddings (model dimension).
        head_dim (int): The dimension of each attention head.
        attention (nn.Module): Custom local attention module.
        linear_query, linear_key, linear_value (nn.Linear): Linear transformation layers for the query, key, and value.
        out_proj (nn.Linear): Final linear layer to project the concatenated outputs of all heads.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model

        self.attention = BatchedLocalAttention()

        # Linear layers for query, key, value
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)

        # Final linear layer
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """
        Forward pass of the MultiHeadLocalAttention module.

        :param query: The query tensor of shape (batch_size, seq_len, 1, d_model).
        :param key: The key tensor of shape (batch_size, seq_len, window_size, d_model).
        :param value: The value tensor of shape (batch_size, seq_len, window_size, d_model).
        :return: The output tensor after applying multi-head local attention and linear projection,
                          with shape (batch_size, seq_len, d_model).
        """

        batch_size, seq_len, _, d_model = query.shape

        # Transform query key value for each head and parallelize using PyTorch vectorization
        # I am transforming the tensors from (batch_size x seq_len x window_size x d_model) to
        # (batch_size x num_heads x seq_len x window_size x d_model//num_heads)
        # The local attention will then be computed in parallel for each head resulting
        query = self.linear_query(query).view(batch_size, -1, self.num_heads, 1, self.head_dim).transpose(1, 2)
        key = self.linear_key(key).view(batch_size, self.num_heads, key.shape[1], key.shape[2], self.head_dim)
        value = self.linear_value(value).view(batch_size, self.num_heads, value.shape[1], value.shape[2], self.head_dim)

        # Apply custom attention
        # Reshape multihead attention output of (batch_size x num_heads x seq_len x d_model//num_heads) to
        # (batch_size x seq_len x d_model) which is the classic shape of the attention output
        # hence we achieved same input output shapes but with different attention computations
        attention_output = self.attention(query, key, value).view(batch_size, seq_len, self.d_model)
        attention_output = self.out_proj(attention_output)

        return attention_output