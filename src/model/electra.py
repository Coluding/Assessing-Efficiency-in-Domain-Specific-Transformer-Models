import torch
import torch.nn as nn
from typing import List
from transformers import (
    ElectraForPreTraining,
    AutoTokenizer,
    ElectraForMaskedLM,
    ElectraConfig,
    RobertaForMaskedLM
)
import yaml


class YAMLElectraConfig(ElectraConfig):
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        electra_config = config["electra"]["standard"]
        super().__init__(**electra_config)
        self.layer_norm_eps = float(self.layer_norm_eps)

        custom_config = config["electra"]["custom"]
        for key, value in custom_config.items():
            setattr(self, key, value)


class ElectraModelWrapper(nn.Module):
    """
    Initialize the Electra Model Wrapper.

    :param electra_config: Configuration object or dict for the Electra model.

    """
    def __init__(self, electra_config: YAMLElectraConfig,
                 generator: nn.Module = None,
                 discriminator: nn.Module = None,
                 tokenizer: AutoTokenizer = None):
        super().__init__()
        self.config = electra_config

        if generator is None:
            if self.config.use_pretrained_generator:
                self.generator = ElectraForMaskedLM.from_pretrained("google/electra-base-generator")
            else:
                self.generator = ElectraForMaskedLM(self.config)
        else:
            self.generator = generator

        if discriminator is None:
            if self.config.use_pretrained_discriminator:
                self.discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
            else:
                self.discriminator = ElectraForPreTraining(self.config)
        else:
            self.discriminator = discriminator

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.generator_tokenizer)
        else:
            self.tokenizer = tokenizer

    def _prepare_generator_inputs(self, input: str):
        tokens = self.tokenizer.tokenize(input, add_special_tokens=True)
        inputs = self.tokenizer.encode(input, return_tensors="pt")
        return tokens, inputs

    def _prepare_discriminator_inputs(self, masked_input_tokens: torch.Tensor,
                                      generator_outputs: torch.Tensor):

        replaced_input_tokens = masked_input_tokens.clone()
        mask_indices = torch.where(masked_input_tokens == self.tokenizer.mask_token_id)
        generated_tokens = torch.argmax(generator_outputs[mask_indices], dim=-1)
        replaced_input_tokens[mask_indices] = generated_tokens

        return replaced_input_tokens

    def _prepare_discriminator_labels(self, input_tokens: torch.Tensor):
        labels = torch.zeros_like(input_tokens)
        labels[torch.where(input_tokens == self.tokenizer.mask_token_id)] = 1
        return labels

    def _prepare_generator_labels(self, input_tokens: torch.Tensor):
        labelled_tokens = input_tokens.clone()
        non_mask_indices = torch.where(input_tokens != self.tokenizer.mask_token_id)
        labelled_tokens[non_mask_indices] = -100
        return labelled_tokens

    def _compute_loss(self, discriminator_loss: torch.Tensor, generator_loss: torch.Tensor):
        return (self.config.loss_weights[1] * discriminator_loss) + (self.config.loss_weights[0] * generator_loss)

    def run_batch_of_text(self, input: str | List[str]):

        generator_tokens, generator_inputs = self._prepare_generator_inputs(input)
        #generator_labels = self._prepare_generator_labels(generator_inputs)
        generator_outputs = self.generator(generator_inputs)

        discriminator_inputs = self._prepare_discriminator_inputs(generator_inputs, generator_outputs)
        discriminator_labels = self._prepare_discriminator_labels(generator_inputs)

        discriminator_outputs = self.discriminator(discriminator_inputs, labels=discriminator_labels)

        return generator_outputs, discriminator_outputs

    def get_discriminator_predictions(self, discriminator_outputs: torch.Tensor):
        return torch.round((torch.sign(discriminator_outputs[1]) + 1) / 2)

    def forward(self, input_tokens: torch.Tensor,
                generator_labels: torch.Tensor = None,
                attention_mask: torch.Tensor = None,):
        """
        Use this when train loader outputs a batch of tokens.

        :param generator_labels:
        :param input_tokens: The tokenized input
        :return:
        """
        generator_outputs = self.generator(input_tokens,
                                           labels=generator_labels,
                                           attention_mask=attention_mask)
        discriminator_inputs = self._prepare_discriminator_inputs(input_tokens, generator_outputs.logits)
        discriminator_labels = self._prepare_discriminator_labels(input_tokens)
        discriminator_outputs = self.discriminator(discriminator_inputs, labels=discriminator_labels)
        loss = self._compute_loss(discriminator_outputs[0], generator_outputs[0])
        return loss, discriminator_outputs, generator_outputs




if __name__ == '__main__':
   # discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    #generator = ElectraForMaskedLM.from_pretrained("google/electra-base-generator")
    #tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    from src.model.custom import YAMLRobertaElectraConfig
    config = ("../config.yml")
    roberta_config = YAMLRobertaElectraConfig("../config.yml")
    generator = RobertaForMaskedLM(roberta_config)
    model = ElectraModelWrapper(config, generator=generator)

    #sentence = "The [MASK] brown fox jumps over the [MASK] dog"

    sentence = "The quick brown [MASK] jumps over the lazy dog"
    fake_sentence = "The quick brown fox fakes over the lazy dog"
    #tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
    #inputs = tokenizer.encode(sentence, return_tensors="pt")

    print(model.run_batch_of_text(sentence))