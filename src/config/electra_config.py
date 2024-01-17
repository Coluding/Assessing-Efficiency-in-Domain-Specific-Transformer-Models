from transformers import PretrainedConfig
import json

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
                 sequence_embeddings: bool = False, relative_position_embeddings: bool = True):
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
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"