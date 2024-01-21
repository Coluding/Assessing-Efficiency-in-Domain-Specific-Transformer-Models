"""Electra model replication"""

import logging
import pdb
import json
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Sequence
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
from tokenizers import Tokenizer
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.activations import get_activation
from transformers.file_utils import ModelOutput
from transformers.models.bert.modeling_bert import BertConfig, BertModel, BertEmbeddings
from transformers.models.reformer.modeling_reformer import ReformerConfig, ReformerModel, ReformerPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
import sys

sys.path.insert(0, "../../src")

from src.model.custom.blocks import ReversibleFullAttentionBlock, AttentionBlock
from src.utils.utils import input_with_timeout

logger = logging.getLogger(__name__)


def _count_parameters(model,
                      max_level: int = 7,
                      debug=False):
    """
    Utility function to display number of parameters
    Inspiration from https://stackoverflow.com/questions/48393608/pytorch-network-parameter-calculation

    :param model: PyTorch model
    :param max_level: Maximum depth level to keep during the aggregation.
    :param debug: Default False. If true, number of parameters for each layers will be displayed
    """
    df = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_name = name.replace(".weight", "").replace(".bias", "").split(".")
            layer_name = {i: layer_name[i] for i in range(len(layer_name))}
            num_param = np.prod(param.size())
            layer_name["params"] = num_param
            df += [layer_name]
            if debug:
                # Display the dimension
                if param.dim() > 1:
                    print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
                else:
                    print(name, ':', num_param)

    df = pd.DataFrame(df)
    # Aggregate layers depending their depth to be easier to read
    df = df.fillna("").groupby([i for i in range(max_level)]).sum()
    pd.set_option('display.max_rows', None)
    print(df)
    print(f"total : {df.params.sum()}")

class DocumentElectraConfig(PretrainedConfig):
    """
        Document Electra config class. Strongly inspired by HuggingFace BertConfig.

    """
    model_type = "document_electra"

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_hidden_layers: int,
                 num_attention_heads: int, intermediate_size: int,
                 max_sentence_length: int, max_sentences: int, max_position_embeddings: int,
                 max_length: int,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.05, attention_probs_dropout_prob: float = 0.05,
                 pad_token_id: int = 0, mask_token_id: int = 1,
                 bos_token_id: int = 2, eos_token_id: int = 3, sep_token_id: int = 4,
                 gradient_checkpointing: bool = False, generator_size: float = 0.25,
                 generator_layer_size: float = 1.0,
                 discriminant_loss_factor: float = 50, mlm_probability: float = 0.15,
                 mlm_replacement_probability: float = 0.85, temperature: float = 1.0,
                 class_output_shape: int = None, regr_output_shape: int = None,
                 fcn_dropout: float = 0.1, chunk_length: int = 128, layer_depth_offset: int = -1,
                 initializer_range: float = 0.02,
                 sequence_embeddings: bool = False, relative_position_embeddings: bool = True,
                 **kwargs):
        super().__init__()
        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings
        self.layer_depth_offset = layer_depth_offset
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.return_dict = True
        self.torchscript = False

        self.chunk_length = chunk_length
        self.mask_token_id = mask_token_id
        self.sep_token_id = sep_token_id

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        self.max_position_embeddings = max_position_embeddings if max_position_embeddings \
            else max_sentence_length * max_sentences
        self.max_sentences = max_sentences
        self.max_sentence_length = max_sentence_length
        self.max_length = max_length
        assert self.max_position_embeddings >= self.max_length

        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.gradient_checkpointing = gradient_checkpointing

        self.initializer_range = initializer_range

        # Downstream task specifics
        self.is_downstream_task = not (class_output_shape is None and regr_output_shape is None)
        if self.is_downstream_task:
            self.is_regression = regr_output_shape is not None and regr_output_shape > 0

            self.num_classes = class_output_shape
            assert self.num_classes is not None and self.num_classes > 0

            self.fcn_dropout = fcn_dropout
            assert self.fcn_dropout is not None

        # Pretraining task specifics
        if not self.is_downstream_task:
            self.mlm_probability = mlm_probability
            self.mlm_replacement_probability = mlm_replacement_probability
            self.temperature = temperature
            self.generator_size = generator_size
            self.generator_layer_size = generator_layer_size if generator_layer_size else generator_size
            self.discriminant_loss_factor = discriminant_loss_factor

    def to_json_string(self, use_diff: bool = False) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """

        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class MyReformerConfig(DocumentElectraConfig):
    def __init__(self, vocab_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 num_hidden_layers: int,
                 num_attention_heads: int,
                 intermediate_size: int,
                 max_sentence_length: int,
                 max_sentences: int,
                 max_position_embeddings: int,
                 max_length: int,
                 attn_layers: Sequence[str] = ("local", "lsh", "local", "lsh", "local", "lsh"),
                 axial_norm_std=1.0, axial_pos_embds=True,
                 axial_pos_shape=[16, 8],
                 axial_pos_embds_dim=[64, 192],
                 chunk_size_lm_head=0,
                 feed_forward_size=512,
                 hash_seed=None,
                 is_decoder=False,
                 layer_norm_eps=1e-12,
                 local_num_chunks_before=1,
                 local_num_chunks_after=0,
                 local_attention_probs_dropout_prob=0.05,
                 local_attn_chunk_length=64,
                 lsh_attn_chunk_length=64,
                 lsh_attention_probs_dropout_prob=0.0,
                 lsh_num_chunks_before=1,
                 lsh_num_chunks_after=0,
                 num_buckets=None,
                 num_hashes=1,
                 tie_word_embeddings=False,
                 use_cache=True,
                 classifier_dropout=None,
                 hidden_act: str = "gelu",
                 hidden_dropout_prob: float = 0.05,
                 attention_probs_dropout_prob: float = 0.05,
                 pad_token_id: int = 0,
                 mask_token_id: int = 1,
                 bos_token_id: int = 2,
                 eos_token_id: int = 3,
                 sep_token_id: int = 4,
                 gradient_checkpointing: bool = False,
                 generator_size: float = 0.25,
                 generator_layer_size: float = 1.0,
                 discriminant_loss_factor: float = 50,
                 mlm_probability: float = 0.15,
                 mlm_replacement_probability: float = 0.85,
                 temperature: float = 1.0,
                 class_output_shape: int = None,
                 regr_output_shape: int = None,
                 fcn_dropout: float = 0.1,
                 chunk_length: int = 128,
                 layer_depth_offset: int = -1,
                 initializer_range: float = 0.02,
                 sequence_embeddings: bool = False,
                 relative_position_embeddings: bool = True,
                 **kwargs):
        super().__init__(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers, num_attention_heads=num_attention_heads,
                         intermediate_size=intermediate_size, max_sentence_length=max_sentence_length,
                         max_sentences=max_sentences, max_position_embeddings=max_position_embeddings,
                         max_length=max_length, hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob,
                         attention_probs_dropout_prob=attention_probs_dropout_prob, pad_token_id=pad_token_id,
                         mask_token_id=mask_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         sep_token_id=sep_token_id, gradient_checkpointing=gradient_checkpointing,
                         generator_size=generator_size, generator_layer_size=generator_layer_size,
                         discriminant_loss_factor=discriminant_loss_factor, mlm_probability=mlm_probability,
                         mlm_replacement_probability=mlm_replacement_probability, temperature=temperature,
                         class_output_shape=class_output_shape, regr_output_shape=regr_output_shape,
                         fcn_dropout=fcn_dropout, chunk_length=chunk_length, layer_depth_offset=layer_depth_offset,
                         initializer_range=initializer_range, sequence_embeddings=sequence_embeddings,
                         relative_position_embeddings=relative_position_embeddings)

        self.attn_layers = attn_layers
        if len(attn_layers) < num_hidden_layers:
            message = (f"Number of attention layers must match number of hidden layers. \n"
                       f"Number of attention layers: {num_hidden_layers}. Number of specified layers: {len(attn_layers)}\n")
            logger.info(message)
            logger.info("Constructing missing layers...")
            num_missing_layers = num_hidden_layers - len(attn_layers)
            last_layer = attn_layers[-1]
            self.attn_layers = list(attn_layers)
            for _ in range(num_missing_layers):
                last_layer = "lsh" if last_layer == "local" else "local"
                self.attn_layers.append(last_layer)

        elif len(attn_layers) > num_hidden_layers:
            logger.warning("Number of attention layers exceeds number of hidden layers. "
                            "This is not a problem but you might want to check your config.")
            self.num_hidden_layers = len(attn_layers)

        self.axial_norm_std = axial_norm_std
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = axial_pos_shape
        self.axial_pos_embds_dim = axial_pos_embds_dim if sum(axial_pos_embds_dim) == self.hidden_size\
            else [round(self.hidden_size * 1/6), round(self.hidden_size * 5/6)]
        self.chunk_size_lm_head = chunk_size_lm_head
        self.feed_forward_size = feed_forward_size
        self.hash_seed = hash_seed
        self.is_decoder = is_decoder
        self.layer_norm_eps = layer_norm_eps
        self.local_num_chunks_before = local_num_chunks_before
        self.local_num_chunks_after = local_num_chunks_after
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.local_attn_chunk_length = local_attn_chunk_length
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class ReversibleDilatedElectraConfig(DocumentElectraConfig):
    model_type = "reversible_dilated_electra"

    def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int, num_hidden_layers: int,
                 num_attention_heads: int, intermediate_size: int,
                 max_sentence_length: int, max_sentences: int, max_position_embeddings: int,
                 max_length: int, dilated: bool = False,
                 hidden_act: str = "gelu", segment_length: Tuple[int] = (128, 128, 128),
                 dilation_rates: Tuple[int] = (1, 3, 5), reversible: bool = True,
                 hidden_dropout_prob: float = 0.05, attention_probs_dropout_prob: float = 0.05,
                 pad_token_id: int = 0, mask_token_id: int = 1,
                 bos_token_id: int = 2, eos_token_id: int = 3, sep_token_id: int = 4,
                 gradient_checkpointing: bool = False, generator_size: float = 0.25,
                 generator_layer_size: float = 1.0,
                 discriminant_loss_factor: float = 50, mlm_probability: float = 0.15,
                 mlm_replacement_probability: float = 0.85, temperature: float = 1.0,
                 class_output_shape: int = None, regr_output_shape: int = None,
                 fcn_dropout: float = 0.1, chunk_length: int = 128, layer_depth_offset: int = -1,
                 initializer_range: float = 0.02,
                 sequence_embeddings: bool = False, relative_position_embeddings: bool = True):
        super().__init__(vocab_size=vocab_size, embedding_size=embedding_size, hidden_size=hidden_size,
                         num_hidden_layers=num_hidden_layers,num_attention_heads=num_attention_heads,
                         intermediate_size=intermediate_size,max_sentence_length=max_sentence_length,
                         max_sentences=max_sentences, max_position_embeddings=max_position_embeddings,
                         max_length=max_length, hidden_act=hidden_act, hidden_dropout_prob=hidden_dropout_prob,
                         attention_probs_dropout_prob=attention_probs_dropout_prob, pad_token_id=pad_token_id,
                         mask_token_id=mask_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                         sep_token_id=sep_token_id, gradient_checkpointing=gradient_checkpointing,
                         generator_size=generator_size, generator_layer_size=generator_layer_size,
                         discriminant_loss_factor=discriminant_loss_factor, mlm_probability=mlm_probability,
                         mlm_replacement_probability=mlm_replacement_probability, temperature=temperature,
                         class_output_shape=class_output_shape, regr_output_shape=regr_output_shape,
                         fcn_dropout=fcn_dropout, chunk_length=chunk_length, layer_depth_offset=layer_depth_offset,
                         initializer_range=initializer_range, sequence_embeddings=sequence_embeddings,
                         relative_position_embeddings=relative_position_embeddings)

        self.segment_length = segment_length if segment_length else (max_length, max_length,
                                                                     max_length)
        self.dilation_rates = dilation_rates
        self.reversible = reversible
        self.dilated = dilated


class ReversibleDilatedElectraEncoder(PreTrainedModel):
    def __init__(self,
                 config: ReversibleDilatedElectraConfig,
                 embedding_size: int,
                 max_relative_position_ids: int,
                 max_sentence_ids: int,
                 relative_position_embeddings: bool = None,
                 sequence_embeddings: bool = None):
        super().__init__(config)

        self.config = config
        bert_config = BertConfig(hidden_act=config.hidden_act,
                                 hidden_dropout_prob=config.hidden_dropout_prob,
                                 attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                 vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                                 feed_forward_size=config.intermediate_size,
                                 intermediate_size=config.intermediate_size,
                                 num_hidden_layers=config.num_hidden_layers,
                                 num_attention_heads=config.num_attention_heads,
                                 max_position_embeddings=config.max_position_embeddings + 1,
                                 chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                 bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                 sep_token_id=config.sep_token_id,
                                 initializer_range=config.initializer_range,
                                 output_hidden_states=True,
                                 return_dict=True,
                                 gradient_checkpointing=config.gradient_checkpointing,
                                 torchscript=False)

        self.input_embeddings = BertEmbeddings(bert_config)
        if config.dilated:
            self.encoder = nn.Sequential(*[AttentionBlock(config.hidden_size,
                                                          config.num_attention_heads,
                                                          config.dilation_rates,
                                                          config.segment_length,
                                                          config.hidden_dropout_prob,
                                                          config.reversible,
                                                          activation=config.hidden_act)
                                         for _ in range(config.num_hidden_layers)])

        else:
            self.encoder = nn.Sequential(*[ReversibleFullAttentionBlock(config.hidden_size,
                                                                        config.num_attention_heads,
                                                                        config.hidden_dropout_prob,
                                                                        config.reversible,
                                                                        activation=config.hidden_act)
                                         for _ in range(config.num_hidden_layers)])


        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings

        assert self.relative_position_embeddings is not None
        assert self.sequence_embeddings is not None
        if self.relative_position_embeddings:
            self.relative_position_embeddings = nn.Embedding(max_relative_position_ids, embedding_size)
        if self.sequence_embeddings:
            self.sequence_embeddings = nn.Embedding(max_sentence_ids, embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, embedding_size)

        if embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            sequence_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None) -> BaseModelOutputWithPooling:
        """

        :param input_ids:
        :param attention_mask:
        :param position_ids:
        :param sequence_ids:
        :param head_mask:
        :param inputs_embeds:
        :param output_hidden_states:
        :param output_attentions:
        :param return_dict:
        :return:
        """
        assert (input_ids is None and inputs_embeds is not None) | (input_ids is not None and inputs_embeds is None)

        assert position_ids is not None
        assert sequence_ids is not None

        # No attention on padding
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = inputs_embeds.byte().any(-1).ne(self.config.pad_token_id)
            assert len(attention_mask.shape) == 2

        assert input_ids is not None
        inputs_embeds = self.token_embeddings.forward(input=input_ids)
        if self.relative_position_embeddings:
            inputs_embeds += self.relative_position_embeddings.forward(input=position_ids)
        if self.sequence_embeddings:
            inputs_embeds += self.sequence_embeddings.forward(input=sequence_ids)

        if hasattr(self, "embeddings_project") and (inputs_embeds is not None):
            inputs_embeds = self.embeddings_project.forward(inputs_embeds)

        input_embeddings: torch.Tensor = self.input_embeddings.forward(input_ids=None,
                                                         token_type_ids=None,
                                                         position_ids=None,
                                                         inputs_embeds=inputs_embeds,
                                                         past_key_values_length=0,
                                                         )


        encoder_outputs = self.encoder.forward(input_embeddings)

        return BaseModelOutputWithPooling(
            last_hidden_state=encoder_outputs,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )


class MyTransformerModel(BertPreTrainedModel):
    """
        Decorator on top of Transformer model to integrate own config logic.
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: BertConfig,
                 embedding_size: int,
                 max_relative_position_ids: int,
                 max_sentence_ids: int,
                 relative_position_embeddings: bool = None,
                 sequence_embeddings: bool = None):
        super().__init__(config)
        self.config = config

        self.transformer = BertModel(config, add_pooling_layer=False)
        # noinspection PyTypeChecker
        self.transformer.set_input_embeddings(None)  # We use our own token embeddings

        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings

        assert self.relative_position_embeddings is not None
        assert self.sequence_embeddings is not None
        if self.relative_position_embeddings:
            self.relative_position_embeddings = nn.Embedding(max_relative_position_ids, embedding_size)
        if self.sequence_embeddings:
            self.sequence_embeddings = nn.Embedding(max_sentence_ids, embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, embedding_size)

        if embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, config.hidden_size)

        self.init_weights()



    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            sequence_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None) -> BaseModelOutputWithPooling:
        """

        :param input_ids:
        :param attention_mask:
        :param position_ids:
        :param sequence_ids:
        :param head_mask:
        :param inputs_embeds:
        :param output_hidden_states:
        :param output_attentions:
        :param return_dict:
        :return:
        """
        assert (input_ids is None and inputs_embeds is not None) | (input_ids is not None and inputs_embeds is None)

        assert position_ids is not None
        assert sequence_ids is not None

        # No attention on padding
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = inputs_embeds.byte().any(-1).ne(self.config.pad_token_id)
            assert len(attention_mask.shape) == 2

        assert input_ids is not None
        inputs_embeds = self.token_embeddings.forward(input=input_ids)
        if self.relative_position_embeddings:
            inputs_embeds += self.relative_position_embeddings.forward(input=position_ids)
        if self.sequence_embeddings:
            inputs_embeds += self.sequence_embeddings.forward(input=sequence_ids)

        if hasattr(self, "embeddings_project") and (inputs_embeds is not None):
            inputs_embeds = self.embeddings_project.forward(inputs_embeds)

        # noinspection PyArgumentEqualDefault
        return self.transformer.forward(
                input_ids=None,  # We use our own config
                position_ids=None,  # The model will also add absolute position embeddings,
                token_type_ids=None,  # We use our own
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds, head_mask=head_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions, return_dict=return_dict)


class MyReformerEncoder(ReformerPreTrainedModel):
    def __init__(self,
                 config: MyReformerConfig,
                 embedding_size: int,
                 max_relative_position_ids: int,
                 max_sentence_ids: int,
                 relative_position_embeddings: bool = None,
                 sequence_embeddings: bool = None
                 ):
        self.config = config

        reformer_config: ReformerConfig = ReformerConfig(hidden_act=config.hidden_act,
                                                         attn_layers=config.attn_layers,
                                                         axial_norm_std=config.axial_norm_std,
                                                         axial_pos_embds=config.axial_pos_embds,
                                                         axial_pos_shape=config.axial_pos_shape,
                                                         axial_pos_embds_dim=config.axial_pos_embds_dim,
                                                         chunk_size_lm_head=config.chunk_size_lm_head,
                                                         eos_token_id=config.eos_token_id,
                                                         feed_forward_size=config.feed_forward_size,
                                                         hash_seed=config.hash_seed,
                                                         hidden_dropout_prob=config.hidden_dropout_prob,
                                                         hidden_size=config.hidden_size,
                                                         attention_head_size=config.hidden_size // config.num_attention_heads,
                                                         initializer_range=config.initializer_range,
                                                         is_decoder=config.is_decoder,
                                                         layer_norm_eps=config.layer_norm_eps,
                                                         local_num_chunks_after=config.local_num_chunks_after,
                                                         local_num_chunks_before=config.local_num_chunks_before,
                                                         local_attention_probs_dropout_prob=config.local_attention_probs_dropout_prob,
                                                         local_attn_chunk_length=config.local_attn_chunk_length,
                                                         lsh_attn_chunk_length=config.lsh_attn_chunk_length,
                                                         lsh_attention_probs_dropout_prob=config.lsh_attention_probs_dropout_prob,
                                                         lsh_num_chunks_after=config.lsh_num_chunks_after,
                                                         lsh_num_chunks_before=config.lsh_num_chunks_before,
                                                         num_attention_heads=config.num_attention_heads,
                                                         num_buckets=config.num_buckets,
                                                         num_hashes=config.num_hashes,
                                                         use_cache=config.use_cache,
                                                         tie_word_embeddings=config.tie_word_embeddings,
                                                         vocab_size=config.vocab_size,
                                                         )

        super().__init__(reformer_config)

        self.transformer = ReformerModel(reformer_config)
        # noinspection PyTypeChecker
        self.transformer.set_input_embeddings(None)

        self.sequence_embeddings = sequence_embeddings
        self.relative_position_embeddings = relative_position_embeddings

        assert self.relative_position_embeddings is not None
        assert self.sequence_embeddings is not None
        if self.relative_position_embeddings:
            self.relative_position_embeddings = nn.Embedding(max_relative_position_ids, embedding_size)
        if self.sequence_embeddings:
            self.sequence_embeddings = nn.Embedding(max_sentence_ids, embedding_size)
        self.token_embeddings = nn.Embedding(config.vocab_size, embedding_size)

        if embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(embedding_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            sequence_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_hidden_states=None,
            output_attentions=None,
            return_dict=None) -> BaseModelOutputWithPooling:
        """

        :param input_ids:
        :param attention_mask:
        :param position_ids:
        :param sequence_ids:
        :param head_mask:
        :param inputs_embeds:
        :param output_hidden_states:
        :param output_attentions:
        :param return_dict:
        :return:
        """
        assert (input_ids is None and inputs_embeds is not None) | (input_ids is not None and inputs_embeds is None)

        assert position_ids is not None
        assert sequence_ids is not None

        # No attention on padding
        if attention_mask is None:
            if input_ids is not None:
                attention_mask = input_ids.ne(self.config.pad_token_id)
            else:
                attention_mask = inputs_embeds.byte().any(-1).ne(self.config.pad_token_id)
            assert len(attention_mask.shape) == 2

        assert input_ids is not None
        inputs_embeds = self.token_embeddings.forward(input=input_ids)
        if self.relative_position_embeddings:
            inputs_embeds += self.relative_position_embeddings.forward(input=position_ids)
        if self.sequence_embeddings:
            inputs_embeds += self.sequence_embeddings.forward(input=sequence_ids)

        if hasattr(self, "embeddings_project") and (inputs_embeds is not None):
            inputs_embeds = self.embeddings_project.forward(inputs_embeds)

        # noinspection PyArgumentEqualDefault
        return self.transformer.forward(
            input_ids=None,  # We use our own config
            position_ids=None,  # The model will also add absolute position embeddings,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds, head_mask=head_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions, return_dict=return_dict)



class DocumentElectraPreTrainedModel(PreTrainedModel, torch.nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    config_class = DocumentElectraConfig
    config: DocumentElectraConfig  # Force the right type
    base_model_prefix = "document_electra"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """

        :param pretrained_model_name_or_path:
        :param model_args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    # Copied from transformers.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module: torch.nn.Module):
        """ Initialize the weights """
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _pad_inputs(self, input_ids, position_ids, sequence_ids, sequence_length):
        if sequence_length % self.config.chunk_length != 0:
            padding = (sequence_length // self.config.chunk_length + 1) * self.config.chunk_length
            padding -= sequence_length
            input_ids = torch.nn.functional.pad(input=input_ids,
                                                pad=[0, padding],
                                                mode='constant',
                                                value=self.config.pad_token_id)
            position_ids = torch.nn.functional.pad(input=position_ids,
                                                   pad=[0, padding],
                                                   mode='constant',
                                                   value=self.config.pad_token_id)
            sequence_ids = torch.nn.functional.pad(input=sequence_ids,
                                                   pad=[0, padding],
                                                   mode='constant',
                                                   value=self.config.pad_token_id)
        return input_ids, position_ids, sequence_ids

    def _pad_embeddings(self, sentence_embeddings):
        if sentence_embeddings.shape[1] % self.config.chunk_length != 0:
            padding = (sentence_embeddings.shape[1] // self.config.chunk_length + 1) * self.config.chunk_length
            padding -= sentence_embeddings.shape[1]
            sentence_embeddings = torch.nn.functional.pad(input=sentence_embeddings,
                                                          pad=[0, 0,
                                                               0, padding],
                                                          mode='constant',
                                                          value=self.config.pad_token_id)
        return sentence_embeddings


@dataclass
class DocumentElectraModelModelOutput:
    """
        Output for the DocumentElectraModel
    """
    pretraining_word_embeddings: torch.Tensor
    downstream_word_embeddings: torch.Tensor
    pretraining_sentence_embeddings: Optional[torch.Tensor] = None
    downstream_sentence_embeddings: Optional[torch.Tensor] = None


# noinspection PyAbstractClass
class DocumentElectraModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Base model.

        2 hierarchical encoders: one to get word level embeddings, and a second one to use the BOS embeddings as input_ids
        and to get a new sentence level embeddings.

        Strongly inspired by HuggingFace library.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        # Hierarchical encoder
        config_word_encoder = BertConfig(hidden_act=config.hidden_act,
                                         hidden_dropout_prob=config.hidden_dropout_prob,
                                         attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                         vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                                         feed_forward_size=config.intermediate_size,
                                         intermediate_size=config.intermediate_size,
                                         num_hidden_layers=config.num_hidden_layers,
                                         num_attention_heads=config.num_attention_heads,
                                         max_position_embeddings=config.max_position_embeddings + 1,
                                         chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                         bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                         sep_token_id=config.sep_token_id,
                                         initializer_range=config.initializer_range,
                                         output_hidden_states=True,
                                         return_dict=True,
                                         gradient_checkpointing=config.gradient_checkpointing,
                                         torchscript=False)

        self.word_encoder = MyTransformerModel(config=config_word_encoder,
                                               embedding_size=config.embedding_size,
                                               relative_position_embeddings=config.relative_position_embeddings,
                                               sequence_embeddings=config.sequence_embeddings,
                                               max_relative_position_ids=config.max_sentence_length + 1,
                                               max_sentence_ids=config.max_sentences + 1)
        hidden_size = config.hidden_size
        self.word_encoder_final_linear = nn.Linear(hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_length = input_ids.shape[0], input_ids.shape[1]

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        input_ids, position_ids, sequence_ids = self._pad_inputs(input_ids, position_ids, sequence_ids, sequence_length)

        # prediction_scores at word level
        # outputs are: batch_size x seq_length x embedding_dimension
        # so this is the bert encoder output
        outputs = self.word_encoder.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids)

        ###
        #   Embedding at token levels
        ###

        # Compute config for the pretraining task at token level
        # slice by sequence length so we only take the non padded tokens into consideration
        pretraining_word_embeddings = self.word_encoder_final_linear.forward(
            input=outputs.last_hidden_state[:, :sequence_length, :])
        assert pretraining_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
            pretraining_word_embeddings.shape

        # Retrieve the config for the downstream task at token level
        # by using the config to specify the layer depth we can retrieve the hidden states for the downstream task
        # at the right layer
        downstream_word_embeddings = outputs.last_hidden_state[:, :sequence_length, :]
        if isinstance(self.config, MyReformerConfig):
            assert downstream_word_embeddings.shape == (num_docs, sequence_length, 2 * self.config.hidden_size), \
                (downstream_word_embeddings.shape, sequence_length)
        else:
            assert downstream_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
                (downstream_word_embeddings.shape, sequence_length)

        return DocumentElectraModelModelOutput(pretraining_word_embeddings=pretraining_word_embeddings,
                                               downstream_word_embeddings=downstream_word_embeddings)


    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A module mapping vocabulary to hidden states.
        """
        self.word_encoder.set_input_embeddings(value)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


class MyReformerDocumentElectraModel(DocumentElectraModel):

    def __init__(self, config: MyReformerConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)
        config_word_encoder = ReformerConfig(hidden_act=config.hidden_act,
                                             hidden_dropout_prob=config.hidden_dropout_prob,
                                             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                             vocab_size=config.vocab_size, hidden_size=config.hidden_size,
                                             feed_forward_size=config.intermediate_size,
                                             intermediate_size=config.intermediate_size,
                                             num_hidden_layers=config.num_hidden_layers,
                                             num_attention_heads=config.num_attention_heads,
                                             max_position_embeddings=config.max_position_embeddings + 1,
                                             chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                             bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                             sep_token_id=config.sep_token_id,
                                             initializer_range=config.initializer_range,
                                             output_hidden_states=True,
                                             return_dict=True,
                                             gradient_checkpointing=config.gradient_checkpointing,
                                             torchscript=False)

        self.word_encoder = MyReformerEncoder(config=config,
                                              embedding_size=config.embedding_size,
                                              relative_position_embeddings=config.relative_position_embeddings,
                                              sequence_embeddings=config.sequence_embeddings,
                                              max_relative_position_ids=config.max_sentence_length + 1,
                                              max_sentence_ids=config.max_sentences + 1)
        hidden_size = config.hidden_size
        # due to the reversible nature of the model, we need to double the hidden size
        self.word_encoder_final_linear = nn.Linear(2 *hidden_size, config.hidden_size)

        self.init_weights()





class ReversibleDilatedElectraModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Base model.

        2 hierarchical encoders: one to get word level embeddings, and a second one to use the BOS embeddings as input_ids
        and to get a new sentence level embeddings.

        Strongly inspired by HuggingFace library.

    """

    # noinspection PyPep8
    def __init__(self, config: ReversibleDilatedElectraConfig):
        super().__init__(config)

        self.word_encoder = ReversibleDilatedElectraEncoder(config=config,
                                                            embedding_size=config.embedding_size,
                                                            relative_position_embeddings=config.relative_position_embeddings,
                                                            sequence_embeddings=config.sequence_embeddings,
                                                            max_relative_position_ids=config.max_sentence_length + 1,
                                                            max_sentence_ids=config.max_sentences + 1)
        hidden_size = config.hidden_size
        self.word_encoder_final_linear = nn.Linear(hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_length = input_ids.shape[0], input_ids.shape[1]

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        input_ids, position_ids, sequence_ids = self._pad_inputs(input_ids, position_ids, sequence_ids, sequence_length)

        # prediction_scores at word level
        # outputs are: batch_size x seq_length x embedding_dimension
        # so this is the bert encoder output
        outputs = self.word_encoder.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            sequence_ids=sequence_ids)

        ###
        #   Embedding at token levels
        ###

        # Compute config for the pretraining task at token level
        # slice by sequence length so we only take the non padded tokens into consideration
        pretraining_word_embeddings = self.word_encoder_final_linear.forward(
            input=outputs.last_hidden_state[:, :sequence_length, :])
        assert pretraining_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
            pretraining_word_embeddings.shape

        # Retrieve the config for the downstream task at token level
        # by using the config to specify the layer depth we can retrieve the hidden states for the downstream task
        # at the right layer
        downstream_word_embeddings = outputs.last_hidden_state[:, :sequence_length, :]
        assert downstream_word_embeddings.shape == (num_docs, sequence_length, self.config.hidden_size), \
            (downstream_word_embeddings.shape, sequence_length)

        return DocumentElectraModelModelOutput(pretraining_word_embeddings=pretraining_word_embeddings,
                                               downstream_word_embeddings=downstream_word_embeddings)


    def set_input_embeddings(self, value: nn.Module):
        """
        Set model's input embeddings.

        Args:
            value (:obj:`nn.Module`): A module mapping vocabulary to hidden states.
        """
        self.word_encoder.set_input_embeddings(value)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError

@dataclass
class DocumentElectraDiscriminatorModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraDiscriminatorModel
    """
    loss: torch.tensor
    token_level_loss: Optional[float] = None
    is_fake_logits: Optional[torch.Tensor] = None


class DocumentElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # input like config.embedding_size?
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.activation = get_activation(config.hidden_act)
        self.config = config

    def forward(self, discriminator_hidden_states):
        """

        :param discriminator_hidden_states:
        :return:
        """
        assert len(discriminator_hidden_states.shape) == 3, discriminator_hidden_states.shape

        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.dense_prediction(hidden_states)

        return logits


# noinspection PyAbstractClass
class DocumentElectraDiscriminatorModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Discriminator model.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        # Hierarchical encoder
        self.document_electra = DocumentElectraModel(config=config)

        # https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/run_pretraining.py#L190
        self.discriminator_electra = DocumentElectraDiscriminatorPredictions(config)
        self.discriminator_document_electra = DocumentElectraDiscriminatorPredictions(config)

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels_at_token_level=None,
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :param labels_at_token_level:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_lengths = input_ids.shape

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        outputs = self.document_electra.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids)

        # Binary classification task at token level
        is_fake_logits = self.discriminator_electra.forward(
            discriminator_hidden_states=outputs.pretraining_word_embeddings).reshape(num_docs, sequence_lengths)
        assert is_fake_logits.shape == (num_docs, sequence_lengths), is_fake_logits.shape

        unmasked_tokens = labels_at_token_level.ne(-100)

        token_level_loss = self.loss(input=is_fake_logits[unmasked_tokens],
                                     target=labels_at_token_level[unmasked_tokens].float())

        return DocumentElectraDiscriminatorModelOutput(
            loss=token_level_loss,
            token_level_loss=token_level_loss.item(),
            is_fake_logits=is_fake_logits)


    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError

class MyReformerDiscriminator(DocumentElectraDiscriminatorModel):
    def __init__(self, config: MyReformerConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)

        # Hierarchical encoder
        self.document_electra = MyReformerDocumentElectraModel(config=config)

        self.discriminator_electra = DocumentElectraDiscriminatorPredictions(config)
        self.discriminator_document_electra = DocumentElectraDiscriminatorPredictions(config)

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()


class ReverseDilatedElectraDiscriminatorModel(DocumentElectraDiscriminatorModel):
    """
    Decorator on top of DocumentElectraDiscriminatorModel to use a dilated encoder.
    """

    def __init__(self, config: ReversibleDilatedElectraConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)

        self.document_electra = ReversibleDilatedElectraModel(config=config)
        self.discriminator_electra = DocumentElectraDiscriminatorPredictions(config)

        self.loss = torch.nn.BCEWithLogitsLoss()

        self.init_weights()



@dataclass
class DocumentElectraGeneratorModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraGeneratorModel
    """
    loss: torch.tensor
    documents_logits: torch.Tensor = None


class DocumentElectraGeneratorPredictions(nn.Module):
    """Prediction module for the generator, made up of two dense layers + linear layer (unlike HuggingFace)."""

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__()

        hidden_size = int(config.hidden_size * config.generator_size)
        self.dense = nn.Linear(hidden_size, config.embedding_size)
        self.activation = get_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.embedding_size)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, generator_hidden_states):
        """

        :param generator_hidden_states:
        :return:
        """
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return self.generator_lm_head(hidden_states)


# noinspection PyAbstractClass
class DocumentElectraGeneratorModel(DocumentElectraPreTrainedModel):
    """
        Document Electra Generator model.

        Strongly inspired by HuggingFace library.

    """

    # noinspection PyPep8
    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        config_sentence_encoder = BertConfig(hidden_dropout_prob=config.hidden_dropout_prob,
                                             attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                                             vocab_size=config.vocab_size,
                                             hidden_size=int(config.hidden_size * config.generator_size),
                                             feed_forward_size=int(config.intermediate_size *
                                                                   config.generator_size),
                                             intermediate_size=int(config.intermediate_size *
                                                                   config.generator_size),
                                             num_hidden_layers=max(1, int(config.num_hidden_layers *
                                                                          config.generator_layer_size)),
                                             num_attention_heads=max(1,
                                                                     int(config.num_attention_heads *
                                                                         config.generator_size)),
                                             max_position_embeddings=config.max_position_embeddings,
                                             chunk_size_feed_forward=0, pad_token_id=config.pad_token_id,
                                             bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                             sep_token_id=config.sep_token_id,
                                             initializer_range=config.initializer_range,
                                             gradient_checkpointing=config.gradient_checkpointing,
                                             return_dict=True,
                                             torchscript=False)

        self.generator_encoder = MyTransformerModel(config=config_sentence_encoder,
                                                    relative_position_embeddings=config.relative_position_embeddings,
                                                    sequence_embeddings=config.sequence_embeddings,
                                                    max_relative_position_ids=config.max_sentence_length + 1,
                                                    embedding_size=config.embedding_size,
                                                    max_sentence_ids=config.max_sentences + 1)

        self.generator_predictions = DocumentElectraGeneratorPredictions(config)

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels_generator=None,  # (docs, seq)
    ):
        """

        :param input_ids:
        :param position_ids:
        :param sequence_ids:
        :param labels_generator:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        (num_docs, sequence_length) = input_ids.shape

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        assert labels_generator is not None
        assert labels_generator.shape == input_ids.shape

        input_ids, position_ids, sequence_ids = self._pad_inputs(input_ids, position_ids, sequence_ids, sequence_length)

        logits = self.generator_encoder.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids).last_hidden_state

        assert len(logits.shape) == 3
        assert logits.shape[0] == num_docs
        if isinstance(self.config, MyReformerConfig):
            # due to the use of reversible layers the hidden dimension doubles when using the layer norm in the end
            assert logits.shape[2] == 2 * int(self.config.hidden_size * self.config.generator_size), logits.shape
        else:
            assert logits.shape[2] == int(self.config.hidden_size * self.config.generator_size), logits.shape

        logits = self.generator_predictions.forward(logits)
        logits = logits[:, :sequence_length, :]  # Remove padding
        assert logits.shape == (num_docs, sequence_length, self.config.vocab_size)

        loss = self.loss(input=logits.reshape(-1, self.config.vocab_size),
                         target=labels_generator.reshape(-1))

        return DocumentElectraGeneratorModelOutput(loss=loss, documents_logits=logits)

    def get_input_embeddings(self) -> nn.Module:
        """
        Returns the model's input embeddings.

        Returns:
            :obj:`nn.Module`: A torch module mapping vocabulary to hidden states.
        """
        return self.generator_encoder.get_input_embeddings()

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


class MyReformerGeneratorModel(DocumentElectraGeneratorModel):
    def __init__(self, config: MyReformerConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)

        generator_config = copy.deepcopy(config)

        generator_config.hidden_size = int(config.hidden_size * config.generator_size)
        generator_config.feed_forward_size = int(config.intermediate_size *
                                                 config.generator_size)
        generator_config.intermediate_size = int(config.intermediate_size *
                                                 config.generator_size)
        generator_config.num_hidden_layers = max(1, int(config.num_hidden_layers *
                                                        config.generator_layer_size))
        generator_config.num_attention_heads = max(1, int(config.num_attention_heads *
                                                          config.generator_size))
        generator_config.attn_layers = config.attn_layers[:generator_config.num_hidden_layers]

        pos_embds_ratio: float = config.axial_pos_embds_dim[0] / sum(config.axial_pos_embds_dim)
        generator_config.axial_pos_embds_dim = [round(generator_config.hidden_size * pos_embds_ratio),
                                                  round(generator_config.hidden_size * (1 - pos_embds_ratio))]


        self.generator_encoder = MyReformerEncoder(config=generator_config,
                                                   relative_position_embeddings=config.relative_position_embeddings,
                                                   sequence_embeddings=config.sequence_embeddings,
                                                   max_relative_position_ids=config.max_sentence_length + 1,
                                                   embedding_size=config.embedding_size,
                                                   max_sentence_ids=config.max_sentences + 1)


        prediction_config = copy.deepcopy(config)

        # due to the use of reversible layers the hidden dimension doubles when using the layer norm in the end
        prediction_config.hidden_size *= 2
        self.generator_predictions = DocumentElectraGeneratorPredictions(prediction_config)

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()


class ReverseDilatedElectraGeneratorModel(DocumentElectraGeneratorModel):
    """
    Decorator on top of DocumentElectraGeneratorModel to use a dilated encoder.
    """

    def __init__(self, config: ReversibleDilatedElectraConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)

        generator_config = copy.deepcopy(config)

        generator_config.hidden_size = int(config.hidden_size * config.generator_size)
        generator_config.feed_forward_size = int(config.intermediate_size *
                                config.generator_size)
        generator_config.intermediate_size = int(config.intermediate_size *
                                config.generator_size)
        generator_config.num_hidden_layers = max(1, int(config.num_hidden_layers *
                                       config.generator_layer_size))
        generator_config.num_attention_heads = max(1,int(config.num_attention_heads *
                                      config.generator_size))

        self.generator_encoder = ReversibleDilatedElectraEncoder(config=generator_config,
                                                                 embedding_size=config.embedding_size,
                                                                 relative_position_embeddings=config.relative_position_embeddings,
                                                                 sequence_embeddings=config.sequence_embeddings,
                                                                 max_relative_position_ids=config.max_sentence_length + 1,
                                                                 max_sentence_ids=config.max_sentences + 1)

        self.generator_predictions = DocumentElectraGeneratorPredictions(config)

        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()


# noinspection PyAbstractClass
class DocumentElectraPretrainingModel(DocumentElectraPreTrainedModel):
    """
        Electra model (with generator) for Electra pretraining task as per Electra paper.

        2 models:
        - A generator
        - A discriminator

        Strongly inspired by HuggingFace library.

    """

    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)

        self.generator = DocumentElectraGeneratorModel(config=self.config)
        self.discriminant = DocumentElectraDiscriminatorModel(config=self.config)

        # Model extension if same size Section 3.2 of Electra paper
        # if self.config.generator_size == 1:
        # Weight sharing between generator and discriminator for input embeddings
        self.generator.generator_encoder.relative_position_embeddings = self.discriminant.document_electra.word_encoder.relative_position_embeddings
        self.generator.generator_encoder.token_embeddings = self.discriminant.document_electra.word_encoder.token_embeddings
        self.generator.generator_encoder.sequence_embeddings = self.discriminant.document_electra.word_encoder.sequence_embeddings

        # Weight sharing between input and output config
        self.generator.generator_predictions.generator_lm_head.weight.data = self.generator.generator_encoder.token_embeddings.weight.data  # .transpose(0,1)

        self.train_generator = True

        self.init_weights()

        _count_parameters(self)

    def _mask_tokens(self, input_ids: torch.Tensor, position_ids: torch.Tensor, sequence_ids: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        """

        assert len(input_ids.shape) == 2  # Batch size (nb of docs) * nb of tokens
        num_docs = input_ids.shape[0]

        assert position_ids.shape == input_ids.shape
        assert sequence_ids.shape == input_ids.shape

        max_sequences = sequence_ids.max()

        # Determine the most corrupted sentences for later use
        probability_matrix = torch.full(size=(num_docs, max_sequences),
                                        device=input_ids.device,
                                        fill_value=0.0)  # max_sequences
        probability_matrix = torch.bernoulli(probability_matrix)
        masked_sentence_ids = probability_matrix.bool().nonzero(as_tuple=False).flatten(start_dim=1)

        labels_at_sentence_level = torch.zeros(size=(num_docs, max_sequences),
                                               device=input_ids.device)
        labels_at_sentence_level[masked_sentence_ids[:, 0], masked_sentence_ids[:, 1]] = 1.

        # Determine the masked tokens (without additional noise from corrupted sentences)
        probability_matrix = torch.full(size=input_ids.shape,
                                        device=input_ids.device,
                                        fill_value=self.config.mlm_probability)

        # Loop of docs and masked sentence ids.
        for mask_idx in masked_sentence_ids:
            probability_matrix[mask_idx[0], sequence_ids[mask_idx[0], :].eq(mask_idx[1] + 1)] = \
                0.0
        masked_tokens = torch.bernoulli(probability_matrix).bool()

        # Create default labels (1 for fake, 0 for original, -100 for padding)
        labels_at_token_level = torch.zeros_like(input_ids)

        # Ignore special tokens
        padding_mask = input_ids.eq(self.config.pad_token_id)  # Padding value
        bos_mask = input_ids.eq(self.config.sep_token_id) | input_ids.eq(self.config.bos_token_id)
        labels_at_token_level[padding_mask | bos_mask] = -100  # We don't compute loss for these
        masked_tokens &= ~(padding_mask | bos_mask)  # And we don't corrupt them

        # Set labels for generator as -100 (ignore) for not masked tokens
        labels_generator = input_ids.clone()
        labels_generator[~masked_tokens] = -100

        # Corrupt the selected inputs with prob
        probability_matrix = torch.full(size=input_ids.shape,
                                        device=input_ids.device,
                                        fill_value=self.config.mlm_replacement_probability)
        replaced_tokens = masked_tokens & torch.bernoulli(probability_matrix).bool()

        generator_input_ids = input_ids.clone()
        generator_input_ids[replaced_tokens] = self.config.mask_token_id

        # Set labels to 1 for tokens to be replaced
        labels_at_token_level[replaced_tokens] = 1

        return (generator_input_ids, labels_generator, masked_tokens, replaced_tokens,
                labels_at_token_level, labels_at_sentence_level)

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            loss_only: bool = True
    ):

        """

        :param loss_only:
        :param sequence_ids:
        :param position_ids:
        :param input_ids:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs = input_ids.shape[0]

        assert position_ids is not None
        assert len(position_ids.shape) == 2

        assert sequence_ids is not None
        assert len(sequence_ids.shape) == 2

        (generator_input_ids, labels_generator, masked_tokens, replaced_tokens,
         labels_at_token_level, labels_at_sentence_level) = self._mask_tokens(input_ids=input_ids,
                                                                              position_ids=position_ids,
                                                                              sequence_ids=sequence_ids)

        assert labels_generator is not None
        assert len(labels_generator.shape) == 2

        assert labels_at_token_level is not None
        assert len(labels_at_token_level.shape) == 2

        assert labels_at_sentence_level is not None
        assert len(labels_at_sentence_level.shape) == 2

        # Generator step
        if self.train_generator:
            generator_outputs = self.generator.forward(input_ids=generator_input_ids,
                                                       position_ids=position_ids,
                                                       sequence_ids=sequence_ids,
                                                       labels_generator=labels_generator)
        else:
            with torch.no_grad():
                generator_outputs = self.generator.forward(input_ids=generator_input_ids,
                                                           position_ids=position_ids,
                                                           sequence_ids=sequence_ids,
                                                           labels_generator=labels_generator)
        mlm_tokens = torch.softmax(generator_outputs.documents_logits, dim=-1).argmax(-1)
        mlm_input_ids = torch.where(replaced_tokens, mlm_tokens, input_ids)
        mlm_input_ids = mlm_input_ids.detach()  # Stop gradients

        # Sampling step
        # Using Gumbel sampling https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        sampled_tokens = torch.nn.functional.gumbel_softmax(logits=generator_outputs.documents_logits,
                                                            tau=self.config.temperature,
                                                            hard=True).argmax(-1)

        assert len(sampled_tokens.shape) == 2  # docs, tokens
        assert sampled_tokens.shape[0] == num_docs
        sampled_input_ids = torch.where(replaced_tokens, sampled_tokens, input_ids)
        sampled_input_ids = sampled_input_ids.detach()  # Stop gradients
        del sampled_tokens

        # Set labels to false when generators give the true values  # the logic should here not before
        labels_at_token_level = torch.zeros_like(input_ids)
        labels_at_token_level[
            sampled_input_ids.ne(input_ids) & labels_generator.ne(-100) & replaced_tokens] = 1  # 0 if original, else 1
        padding_mask = input_ids.eq(self.config.pad_token_id)  # Padding value
        bos_mask = input_ids.eq(self.config.sep_token_id) | input_ids.eq(self.config.bos_token_id)
        labels_at_token_level[padding_mask | bos_mask] = -100  # We don't compute loss for these

        # Discriminant step
        disc_outputs = self.discriminant.forward(input_ids=sampled_input_ids,
                                                 position_ids=position_ids,
                                                 sequence_ids=sequence_ids,
                                                 labels_at_token_level=labels_at_token_level,)

        # Combine losses
        loss = generator_outputs.loss + self.config.discriminant_loss_factor * disc_outputs.loss

        # Prepare outputs. Note: adaptation for PEP8
        output = DocumentElectraPretrainingModelOutput(loss=loss,
                                                       generator_loss=generator_outputs.loss.item(),
                                                       discriminant_loss=disc_outputs.loss.item(),
                                                       discriminant_token_loss=disc_outputs.token_level_loss)
        if not loss_only:
            output.mlm_input_ids = mlm_input_ids
            output.sampled_input_ids = sampled_input_ids
            output.labels_generator = labels_generator
            output.is_fake_logits = disc_outputs.is_fake_logits
            output.labels_at_token_level = labels_at_token_level

        return output

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


class MyReformerElectraPretrainingModel(DocumentElectraPretrainingModel):
    def __init__(self, config: MyReformerConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)
        self.generator = MyReformerGeneratorModel(config=config)
        self.discriminant = MyReformerDiscriminator(config=config)

        self.generator.generator_encoder.relative_position_embeddings = self.discriminant.document_electra.word_encoder.relative_position_embeddings
        self.generator.generator_encoder.token_embeddings = self.discriminant.document_electra.word_encoder.token_embeddings
        self.generator.generator_encoder.sequence_embeddings = self.discriminant.document_electra.word_encoder.sequence_embeddings

        self.generator.generator_predictions.generator_lm_head.weight.data = self.generator.generator_encoder.token_embeddings.weight.data  # .transpose(0,1)

        self.train_generator = True

        self.init_weights()

        _count_parameters(self)


class ReversibleDilatedElectraPretrainingModel(DocumentElectraPretrainingModel):
    def __init__(self, config: ReversibleDilatedElectraConfig):
        DocumentElectraPreTrainedModel.__init__(self, config)

        self.generator = ReverseDilatedElectraGeneratorModel(config=config)
        self.discriminant = ReverseDilatedElectraDiscriminatorModel(config=config)

        self.generator.generator_encoder.relative_position_embeddings = self.discriminant.document_electra.word_encoder.relative_position_embeddings
        self.generator.generator_encoder.token_embeddings = self.discriminant.document_electra.word_encoder.token_embeddings
        self.generator.generator_encoder.sequence_embeddings = self.discriminant.document_electra.word_encoder.sequence_embeddings

        # Weight sharing between input and output config
        self.generator.generator_predictions.generator_lm_head.weight.data = self.generator.generator_encoder.token_embeddings.weight.data  # .transpose(0,1)

        self.train_generator = True

        self.init_weights()

        _count_parameters(self)


@dataclass
class DocumentElectraPretrainingModelOutput(ModelOutput):
    """
        Output class for the DocumentElectraPretrainingModel
    """
    loss: torch.tensor

    generator_loss: float = None
    mlm_input_ids: Optional[torch.Tensor] = None
    sampled_input_ids: Optional[torch.Tensor] = None
    labels_generator: Optional[torch.Tensor] = None

    discriminant_loss: Optional[float] = None
    discriminant_token_loss: Optional[float] = None
    is_fake_logits: Optional[torch.Tensor] = None
    labels_at_token_level: Optional[torch.Tensor] = None


# noinspection PyAbstractClass
class DocumentElectraModelHead(DocumentElectraPreTrainedModel):
    """
        Head on top of DocumentElectra
    """

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError()

    def __init__(self, config: DocumentElectraConfig):
        super().__init__(config)
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = get_activation(config.hidden_act)
        self.linear_2 = nn.Linear(config.hidden_size, config.num_classes)

        self.dropout = nn.Dropout(p=self.config.fcn_dropout)

    def forward(
            self,
            document_embeddings
    ):
        """

        :param document_embeddings:
        :return:
        """
        document_embeddings = self.dropout(document_embeddings)
        document_embeddings = self.linear_1.forward(document_embeddings)
        document_embeddings = self.activation(document_embeddings)
        return self.linear_2.forward(document_embeddings)


# noinspection PyAbstractClass
class DocumentElectraModelForDownstream(DocumentElectraPreTrainedModel):
    """
    Hierarchical Electra model and linear head for downstream task.

    """

    def __init__(self, config: DocumentElectraConfig, tokenizer: Tokenizer = None):
        super().__init__(config)

        self.document_electra = DocumentElectraModel(config=config)
        self.head = DocumentElectraModelHead(config=config)

        self.loss = nn.MSELoss() if self.config.is_regression else nn.CrossEntropyLoss()

        self.tokenizer = tokenizer

        self.init_weights()

        _count_parameters(self)

    def forward(
            self,
            input_ids=None,  # (docs, seq)
            position_ids=None,  # (docs, seq)
            sequence_ids=None,  # (docs, seq)
            labels=None,  # (docs)
    ):
        """

        :param sequence_ids:
        :param position_ids:
        :param input_ids:
        :param labels:
        :return:
        """
        assert input_ids is not None
        assert len(input_ids.shape) == 2
        num_docs, sequence_lengths = input_ids.shape

        assert labels is not None
        assert labels.shape == torch.Size([num_docs]), (labels.shape, num_docs)

        # Document Electra encoder
        outputs = self.document_electra.forward(input_ids=input_ids,
                                                position_ids=position_ids,
                                                sequence_ids=sequence_ids)

        if outputs.downstream_sentence_embeddings is None:
            # Merge all sentences embeddings for a document config
            # Use the config for position ids == 1
            # document_embeddings = []
            # for d in range(num_docs):
            #    document_embeddings += [outputs.downstream_word_embeddings[d, position_ids[d].eq(1), :].mean(0)]
            # document_embeddings = torch.stack(document_embeddings)
            # document_embeddings = outputs.downstream_word_embeddings[:, 0, :]
            non_padding_mask = input_ids.ne(self.config.pad_token_id)
            document_embeddings = (non_padding_mask.unsqueeze(-1) * outputs.pretraining_word_embeddings).sum(
                1)  # Average pooling on all words
            document_embeddings /= non_padding_mask.sum(-1).unsqueeze(-1)
            assert document_embeddings.shape == (num_docs, self.config.hidden_size), document_embeddings.shape

        else:
            # Merge all sentences embeddings for a document config
            # Use the first config like Electra for classification
            # https://github.com/google-research/electra/blob/79111328070e491b287c307906701ebc61091eb2/model/modeling.py#L254
            document_embeddings = outputs.downstream_sentence_embeddings[:, 0, :]
            assert document_embeddings.shape == (num_docs, self.config.hidden_size)

        # Classification head
        documents_logits = self.head.forward(document_embeddings=document_embeddings)
        assert documents_logits.shape == (num_docs, self.config.num_classes), documents_logits.shape

        if self.tokenizer:
            # Debug only
            print(
                self.tokenizer.decode_batch([input_ids[0].cpu().numpy().tolist()], skip_special_tokens=False)[0].split(
                    "<PAD>")[0])
            print(labels[0])
            print(torch.softmax(documents_logits[0], dim=-1).argmax(-1),
                  torch.softmax(documents_logits[0], dim=-1),
                  documents_logits.shape)

        # Compute loss
        if self.config.is_regression and self.config.num_classes == 1:
            labels = labels.reshape(-1, 1)  # Batch size * 1
            assert documents_logits.shape == labels.shape
        loss = self.loss.forward(input=documents_logits, target=labels)

        # If binary class, keep only the logits for class 1
        if not self.config.is_regression and self.config.num_classes == 2:
            documents_logits = documents_logits[:, 1]

        return DocumentElectraModelForClassificationOutput(loss=loss,
                                                           logits=documents_logits)

    def _forward_unimplemented(self, *inputs: Any) -> None:
        raise NotImplementedError


@dataclass
class DocumentElectraModelForClassificationOutput(ModelOutput):
    """
        Output class for DocumentElectraModelForDownstream
    """
    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None


if __name__ == "__main__":
    config = MyReformerConfig(
        vocab_size=1000,
        embedding_size=128,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        max_position_embeddings=512,
        max_sentences=10,
        max_sentence_length=50,
        max_length=512,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        gradient_checkpointing=False,
        initializer_range=0.02,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        sep_token_id=3,
        mask_token_id=4,
        mlm_probability=0.15,
        mlm_replacement_probability=0.8,
        temperature=1.0,
        generator_size=0.25,
        generator_layer_size=0.25,
        discriminant_loss_factor=50.,
        fcn_dropout=0.1,

    )
    model = MyReformerElectraPretrainingModel(config=config).to("cuda")
    inputs = {"input_ids": torch.randint(0, 1000, (10, 64), device="cuda"), "position_ids": torch.randint(0, 9, (10, 64),device="cuda"),
                "sequence_ids": torch.randint(0, 2, (10, 64),device="cuda")}

    out = model(**inputs)
    out.loss.backward()