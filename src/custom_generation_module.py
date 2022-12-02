import re
from typing import Optional, Iterator, Tuple, Dict, List, Callable, Union

import omegaconf
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, ForcedBOSTokenLogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import Seq2SeqLMOutput

from transformers.models.bart import BartForConditionalGeneration

from classy.data.data_drivers import GenerationSample
from classy.pl_modules.base import ClassyPLModule, ClassificationOutput
from classy.pl_modules.mixins.task import (
    GenerationTask,
)
from transformers.generation_utils import GenerationMixin
from transformers.models.bart.modeling_bart import shift_tokens_right


class HFCustomGenerationPLModule(GenerationTask, ClassyPLModule):
    def __init__(
        self,
        transformer_model: str,
        decoding_skip_special_tokens: bool,
        decoding_clean_up_tokenization_spaces: bool,
        optim_conf: omegaconf.DictConfig,
        additional_special_tokens: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(vocabulary=None, optim_conf=optim_conf)
        self.save_hyperparameters()
        self.generative_model = HFCustomGenerativeModel.from_transformer_model(
            transformer_model,
            decoding_skip_special_tokens=decoding_skip_special_tokens,
            decoding_clean_up_tokenization_spaces=decoding_clean_up_tokenization_spaces,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    def load_prediction_params(self, prediction_params: Dict):
        self.generative_model.load_generation_params(prediction_params)

    def forward(self, *args, **kwargs) -> ClassificationOutput:
        return self.generative_model.forward(*args, **kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch)
        self.log("loss", forward_output.loss)
        self.log("ppl", torch.exp(forward_output.loss))
        return forward_output.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log("val_loss", forward_output.loss)
        self.log("val_ppl", torch.exp(forward_output.loss), prog_bar=True, on_step=False, on_epoch=True)
        return forward_output.loss

    def test_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**batch)
        self.log("test_loss", forward_output.loss)
        self.log("test_ppl", torch.exp(forward_output.loss), prog_bar=True, on_step=False, on_epoch=True)
        return forward_output.loss

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[GenerationSample, str]]:
        return self.generative_model.batch_predict(*args, **kwargs)


class HFCustomGenerativeModel(nn.Module):
    @classmethod
    def from_transformer_model(cls, transformer_model: str, **kwargs):
        if re.fullmatch("facebook/bart-(base|large)", transformer_model):
            return BartCustomGenerativeModule(transformer_model, **kwargs)
        else:
            raise ValueError

    def __init__(
        self, transformer_model: str, decoding_skip_special_tokens: bool, decoding_clean_up_tokenization_spaces: bool
    ):
        super().__init__()
        self.generation_params = {}

    def load_generation_params(self, generation_params: Dict):
        self.generation_params = generation_params

    def forward(self, *args, **kwargs) -> ClassificationOutput:
        raise NotImplementedError

    def batch_predict(self, *args, **kwargs) -> Iterator[Tuple[GenerationSample, str]]:
        raise NotImplementedError


class CustomForcedBOSTokenLogitsProcessor:
    r"""
    :class:`~transformers.LogitsProcessor` that enforces the specified token[s] as the first generated token[s].

    Args:
        bos_token_ids (:obj:`int`):
            The id[s] of the token[s] to force as the first generated token[s].
    """

    def __init__(self, bos_token_ids: Union[int, List[int]]):
        self.bos_token_ids = [bos_token_ids] if type(bos_token_ids) != list else bos_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if 1 <= cur_len <= len(self.bos_token_ids):
            curr_forced_token_id = self.bos_token_ids[cur_len-1]
            num_tokens = scores.shape[1]
            scores[:, [i for i in range(num_tokens) if i != curr_forced_token_id]] = -float("inf")
            scores[:, curr_forced_token_id] = 0
        return scores


class BartCustomForConditionalGeneration(BartForConditionalGeneration, GenerationMixin):

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
    ) -> LogitsProcessorList:
        processors = super(BartCustomForConditionalGeneration, self)._get_logits_processor(
            repetition_penalty, no_repeat_ngram_size, encoder_no_repeat_ngram_size, encoder_input_ids, bad_words_ids, min_length, max_length, eos_token_id, forced_bos_token_id, forced_eos_token_id, prefix_allowed_tokens_fn, num_beams, num_beam_groups, diversity_penalty, remove_invalid_values
        )
        change_idx = None
        for i, processor in enumerate(processors):
            if type(processor) == ForcedBOSTokenLogitsProcessor:
                change_idx = i

        if change_idx is not None:
            processors[change_idx] = CustomForcedBOSTokenLogitsProcessor(forced_bos_token_id)

        return processors


class BartCustomGenerativeModule(HFCustomGenerativeModel):
    def __init__(
        self,
        transformer_model: str,
        decoding_skip_special_tokens: bool,
        decoding_clean_up_tokenization_spaces: bool,
        additional_special_tokens: Optional[List[str]] = None,
        compositional_rank: int = None
    ):
        super().__init__(transformer_model, decoding_skip_special_tokens, decoding_clean_up_tokenization_spaces)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=list(additional_special_tokens)
            if additional_special_tokens is not None
            else None,
            use_fast=True,
            add_prefix_space=True,  # todo this should be read from config (like facebook/bart-large-xsum has it False)
        )
        self.model = BartCustomForConditionalGeneration.from_pretrained(transformer_model)
        if additional_special_tokens is not None and len(additional_special_tokens) > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.decoding_skip_special_tokens = decoding_skip_special_tokens
        self.decoding_clean_up_tokenization_spaces = decoding_clean_up_tokenization_spaces
        self.compositional_rank = compositional_rank

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        decoder_attention_mask: Optional[torch.Tensor],
        **kwargs,
    ) -> ClassificationOutput:
        decoder_input_ids = None
        if labels is not None and self.compositional_rank is not None and self.compositional_rank > 0:
            decoder_input_ids = shift_tokens_right(
                labels, self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )
            labels[:, :self.compositional_rank] = -100
        bart_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        return ClassificationOutput(
            loss=bart_out.loss,
            logits=bart_out.logits,
            probabilities=bart_out.logits.softmax(dim=-1),
            predictions=bart_out.logits.argmax(dim=-1),
        )

    def batch_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_start: torch.Tensor,
        num_return_sequences: int = 1,  # todo implement
        **kwargs,
    ) -> Iterator[Tuple[GenerationSample, str]]:
        assert len(set(decoder_start.squeeze(-1).tolist())) == 1

        samples = kwargs.get("samples")
        compositional_tokens = getattr(samples[0], "compositional_tokens", None)
        if compositional_tokens is not None:
            compositional_tokens = [f"<{ct}>" for ct in compositional_tokens]
            compositional_token_ids = [self.tokenizer.convert_tokens_to_ids(ct) for ct in compositional_tokens]
            forced_bos_token_id = compositional_token_ids
        else:
            forced_bos_token_id = self.tokenizer.bos_token_id

        # generate
        bart_out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_return_sequences=num_return_sequences,
            forced_bos_token_id=forced_bos_token_id,
            **self.generation_params,
        )

        # decode
        decoded_bart_out = self.tokenizer.batch_decode(
            bart_out,
            skip_special_tokens=self.decoding_skip_special_tokens,
            clean_up_tokenization_spaces=self.decoding_clean_up_tokenization_spaces,
        )

        # postprocess
        for sample, prediction in zip(samples, decoded_bart_out):
            yield sample, prediction
