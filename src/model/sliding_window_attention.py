import copy

from torch import Tensor

from utils import slice_tensor_in_windows, remove_global_attention_token_create_global_attention_tensor
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from typing import Sequence, Tuple, Union, Optional, Any
import xformers.components
from xformers.components.attention import (
    maybe_sparsify,
    sparsify,
)
from xformers.components.attention.attention_patterns import (
    causal_1d_pattern,
    local_1d_pattern
)
import xformers.ops as xops
from xformers.components.feedforward import MLP
from xformers.components.multi_head_dispatch import MultiHeadDispatch
from enum import Enum

import utils


class QKVProjectionOption(Enum):
    INDIVIDUAL = 1
    QK = 2
    SAME = 3


class AttentionProjector(nn.Module):
    def __init__(self, d_model: int, projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL):
        super().__init__()

        self.projection_option = projection_option
        if projection_option == QKVProjectionOption.INDIVIDUAL:
            self.q_proj = nn.Linear(d_model,d_model)
            self.v_proj = nn.Linear(d_model,d_model)
            self.k_proj = nn.Linear(d_model,d_model)

        elif projection_option == QKVProjectionOption.QK:
            self.qk_proj =nn.Linear(d_model,d_model)
            self.v_proj = nn.Linear(d_model,d_model)

        elif projection_option == QKVProjectionOption.SAME:
            self.proj = nn.Linear(d_model,d_model)

    def forward(self, x: torch.Tensor):
        q,k,v = None, None, None
        if self.projection_option == QKVProjectionOption.INDIVIDUAL:
            q = self.q_proj(x)
            v = self.v_proj(x)
            k = self.v_proj(x)
        elif self.projection_option == QKVProjectionOption.QK:
            q = self.qk_proj(x)
            v = self.v_proj(x)
            k = torch.clone(q)

        elif self.projection_option == QKVProjectionOption.SAME:
            q = self.proj(x)
            v = torch.clone(q)
            k = torch.clone(q)

        return q, k, v


class DilatedAttention(nn.Module):
    """
    Implements a Dilated Attention mechanism in a neural network.

    This attention mechanism allows for efficient computation of self-attention by
    dividing the input sequence into segments and applying different dilation rates
    to different segments. This approach is beneficial for capturing long-range
    dependencies in sequences.

    Parameters:
    segment_lengths (Sequence[int]): A sequence of integers specifying the lengths
                                     of each segment in the input sequence.
    dilation_rates (Sequence[int]): A sequence of integers specifying the dilation
                                    rate for each segment.
    softmax_scale (Optional[float], optional): Scaling factor for the softmax
                                               normalization. Defaults to None.
    attention_dropout (float): Dropout rate for the attention weights. Defaults to 0.0.
    op (Optional[xops.AttentionOp], optional): An optional operation for
                                               memory-efficient attention computation.
                                               Defaults to None.

    Raises:
    ValueError: If the length of segment_lengths and dilation_rates are not equal.

    Note:
    The input tensors to the forward method should have the shape (b, n, h, d), where:
    - b is the batch size
    - n is the sequence length
    - h is the number of attention heads
    - d is the dimensionality of each head.

    """

    def __init__(
        self,
        segment_lengths: Sequence[int],
        dilation_rates: Sequence[int],
        softmax_scale: Optional[float] = None,
        attention_dropout: float = 0.0,
        op: Optional[xops.AttentionOp] = None,
    ):

        super().__init__()
        if len(segment_lengths) != len(dilation_rates):
            raise ValueError(
                "segment_lengths and dilation_rates must have the same length"
            )

        self.segment_lengths = segment_lengths
        self.dilation_rates = dilation_rates
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.op = op

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes output of dilated attention.

        :param query: Query tensor of shape (b,n,h,d)
        :param key: Key tensor of shape (b,n,h,d)
        :param value: Value tensor of shape (b,n,h,d)
        :return: Dilated attention output of shape (b,n,h,d)
        """
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #   s - segment length
        #   r - dilation rate
        #   g - group size (i.e. number of heads per segment length)
        #
        # Input shape of query, key, value: (b, n, h, d)
        b, _, h, _ = query.shape
        out = torch.zeros_like(query)

        # number of different groups based on the number of dilation rates
        num_groups = len(self.dilation_rates)
        group_sizes = [h // num_groups] * num_groups

        for i in range(h % num_groups):
            group_sizes[i] += 1

        for i, (g, r, s) in enumerate(
            zip(group_sizes, self.dilation_rates, self.segment_lengths)
        ):
            # Split the input sequences into segments of length 'self.segment_length'
            # That is why sequence_length % seg_length = 0
            q = rearrange(query, "b (n s) h d -> b n s h d", s=s)
            k = rearrange(key, "b (n s) h d -> b n s h d", s=s)
            v = rearrange(value, "b (n s) h d -> b n s h d", s=s)
            # Apply dilation and segment offset
            offset = i % r
            hmin = i * g
            hmax = (i + 1) * g
            # hmin and hmax are the heads that are used for this dilation rate
            # we split all heads into len(dilation_rates) different groups such that all heads of one group cover one dilation rate
            # Additionally the dilation is done by offset::r in the indexing operation

            q = q[:, :, offset::r, hmin:hmax, :]
            k = k[:, :, offset::r, hmin:hmax, :]
            v = v[:, :, offset::r, hmin:hmax, :]
            # Fold all 'n' segments into the batch dimension
            q = rearrange(q, "b n s h d -> (b n) s h d")
            k = rearrange(k, "b n s h d -> (b n) s h d")
            v = rearrange(v, "b n s h d -> (b n) s h d")

            # Apply memory efficient attention with xformers automatic detection which attention computation is the best: self.op

            x = xops.memory_efficient_attention(
                query=q, key=k, value=v, op=self.op
            )
            # Unfold n segments back out of the batch dimension.
            x = rearrange(x, "(b n) s h d -> b n s h d", b=b)

            # Normalize attention outputs across the sequence length dimension. This
            # is necessary because the attention outputs from each dilation rate /
            # segment length are summed together.
            x = x / x.sum(dim=(1, 2), keepdim=True)

            # Gather the attention outputs from each dilation rate / segment length.
            # We computed segmented attention per segment and head such that out final result is of shape (b n s h d) again
            # h is the number of heads for this group
            #
            out = rearrange(out, "b (n s) h d -> b n s h d", s=s)
            out[:, :, offset::r, hmin:hmax, :] += x
            out = rearrange(out, "b n s h d -> b (n s) h d", s=s)

        # We have already normalized each attention output across the sequence length.
        # Now, normalize across all attention outputs by dividing by the number of
        # attention groups.  See: https://arxiv.org/pdf/2307.02486.pdf, Eq. 10
        return out / num_groups


class MultiheadDilatedAttention(nn.Module):
    """
     Implements a multi-head dilated attention mechanism in a Transformer model.

    This module is designed to provide attention with varying levels of granularity,
    using a combination of multiple heads and dilated attention. Each head in the
    multi-head setup can focus on different segments of the input with varying dilation
    rates, allowing the model to capture both local and global dependencies in the data.

    Parameters:
    d_model (int): The dimensionality of the input and output of the attention module.
    num_heads (int): The number of attention heads.
    dilation_rates (Sequence[int]): A sequence of integers specifying the dilation
                                    rates for each attention head.
    segment_lengths (Sequence[int]): A sequence of integers specifying the lengths of
                                     segments for each attention head.
    dropout (float, optional): Dropout rate applied to the attention scores.
                               Defaults to 0.0.
    bias (bool, optional): If set to True, adds a learnable bias to the projections.
                           Defaults to True.
    layer_norm (bool, optional): If set to True, adds layer normalization. Defaults
                                 to True.
    layer_norm_eps (float, optional): The epsilon value for layer normalization.
                                      Defaults to 1e-5.
    gamma_init (float, optional): Custom gain factor for initialization. Defaults to 1.0.
    device (Optional[Union[torch.device, str]], optional): The device on which the
                                                           module is to be loaded.
    dtype (Optional[torch.dtype], optional): The data type of the module parameters.
    op (Optional[xops.AttentionOp], optional): An optional operation for
                                               memory-efficient attention computation.

    Raises:
    ValueError: If `d_model` is not divisible by `num_heads`, or if `head_dim` (computed
                as `d_model` / `num_heads`) is not divisible by 8 or is greater than 128,
                or if the number of dilation rates does not equal the number of segment
                lengths.

    This class combines multi-head attention with dilated attention, allowing for
    efficient and flexible attention mechanisms over sequences. It is particularly
    useful in scenarios where both local and long-range dependencies are important.

    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dilation_rates: Sequence[int],
        segment_lengths: Sequence[int],
        dropout: float = 0.0,
        projection_option: QKVProjectionOption = QKVProjectionOption.SAME,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        op: Optional[xops.AttentionOp] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if not d_model % self.num_heads == 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"num_heads ({num_heads})"
            )
        num_dilations = len(dilation_rates)
        num_segments = len(segment_lengths)
        if num_dilations != num_segments:
            raise ValueError(
                f"len(dilation_rates) ({num_dilations}) must be equal to "
                f"len(segment_lengths) ({num_segments})"
            )
        head_dim = d_model // num_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        self.q_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )
        self.k_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )
        self.attention = DilatedAttention(
            segment_lengths=segment_lengths,
            dilation_rates=dilation_rates,
            attention_dropout=dropout,
            op=op,
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                d_model, eps=layer_norm_eps, device=device, dtype=dtype
            )
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

        self.in_proj = AttentionProjector(d_model, projection_option)

    def _reset_parameters(self):
        """
        Initializes the weights and biases of the query, key, value, and output projections
        within an attention mechanism of a Transformer model.

        This method employs the Xavier normal initialization for weights, which helps in
        maintaining a consistent scale of gradients, thus making the training of deep
        networks more efficient. For biases, a constant initialization is used.

        Additionally, for the value and output projections, a custom gain factor is applied
        during initialization, following strategies outlined in the MAGNETO paper. This
        custom gain is crucial for adapting the initialization process to the specific
        architectural needs of the Transformer model, potentially improving model
        performance and training stability.

        The initialization process is as follows:
        - Xavier normal initialization for weights of query, key, value, and output
          projections. The value and output projections use a custom gain (`self.gamma_init`).
        - Constant initialization (zero value) for biases of query, key, value, and output
          projections, if they exist.

        The Xavier normal initialization sets the weights from a normal distribution with
        a mean of 0 and a variance based on the number of input and output neurons. This
        approach aims to keep the scale of gradients roughly the same in all layers,
        thereby aiding in the efficient training of the network. The custom gain factor
        for the value and output projections is a scaling factor that adjusts the variance
        of the initialization, typically based on specific aspects of the network's
        architecture.

        When using larger models, the gamma should be provided as keyword argument since it requires knowledge about
        the number of encoder layers.

        For reference see here: https://arxiv.org/pdf/2210.06423.pdf

        :return: None
        """
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        """
        Computes the forward pass of the dilated attention model. Returns the multihead attention output.

        :param query: Query tensor of shape (b,n,h,d//h)
        :param key: Key tensor of shape (b,n,h,d//h)
        :param value: Value tensor of shape (b,n,h,d//h)
        :return: Attention output of shape (b,n,d)
        """
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q,k,v = self.in_proj(x)
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.num_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x = self.attention(q, k, v)

        # combine heads again
        x = rearrange(x, "b n h d -> b n (h d)")

        # LongNet follows MAGNETO archtiecture where an extra layer norm is applied before linear output projection
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)

        # Linear projection to account for the split up of the heads
        x = self.out_proj(x)

        return x


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


################################# Deprecated ############################################


class MultiHeadDilatedLocalAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, window_size: int,
                 global_tokens: Optional[torch.Tensor] = None,
                 dilation_rate: Optional[int] = None,
                 projection_option: QKVProjectionOption = QKVProjectionOption.INDIVIDUAL):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.global_indices = global_tokens
        self.dilation_rate = dilation_rate

        self.attention_projector = AttentionProjector(d_model, projection_option)
        local_attention = LocalAttentionDilation(window_size=window_size)

        if dilation_rate is not None:
            local_attention.set_dilation(dilation_rate)

        if global_tokens is not None:
            local_attention.set_global_indices(global_tokens)

        self.multihead = MultiHeadDispatch(attention=local_attention, dim_model=d_model, num_heads=num_heads)

    def forward(self, x: torch.Tensor):
        q, k, v = self.attention_projector(x)
        return self.multihead(q,k,v)


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
        """

        # also sets loca_token_indices attribute
        global_keys, sliding_window_key = remove_global_attention_token_create_global_attention_tensor(key,
                                                                      self.global_token_indices,
                                                                      self)

        global_values, sliding_window_values = remove_global_attention_token_create_global_attention_tensor(value,
                                                                                                      self.global_token_indices)

        global_queries, sliding_window_queries = remove_global_attention_token_create_global_attention_tensor(query,
                                                                                                            self.global_token_indices)

        sliding_window_value = slice_tensor_in_windows(sliding_window_values, self.window_size, self.dilation_rate)
        sliding_window_query = slice_tensor_in_windows(sliding_window_queries, self.window_size, self.dilation_rate)


        return (global_keys, sliding_window_key), (global_queries, sliding_window_query), (global_values, sliding_window_value)

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

        batch_size, seq_len, d_model = query.shape

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



if __name__ == "__main__":
    b,n,d = (128,4096,512)
    h = 8
    mh = MultiheadDilatedAttention(d_model=d, num_heads=h, dilation_rates=[1,3,5], segment_lengths=[128,512,1024], dtype=torch.float32).to("cuda")
    x = torch.randn(b,n, d,dtype=torch.float32).to("cuda")
    o = mh(x)
    o[0]