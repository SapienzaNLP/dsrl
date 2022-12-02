import collections
import re
from typing import Optional, Callable, Iterable, Dict, Any, Iterator, Union, List, Generator

import torch
from classy.data.dataset.hf.generation import EncDecHFGenerationSampleEncoder
from transformers import AutoTokenizer

from classy.data.data_drivers import SequenceSample, TokensSample, SentencePairSample, QASample, GenerationSample
from classy.data.dataset.base import batchify, BaseDataset
from classy.utils.vocabulary import Vocabulary


class HFCustomGenerationDataset(BaseDataset):
    @staticmethod
    def requires_vocab() -> bool:
        return False

    @staticmethod
    def fit_vocabulary(samples: Iterator[Union[SentencePairSample, SequenceSample, TokensSample]]) -> Vocabulary:
        raise NotImplementedError

    def __init__(
        self,
        samples_iterator: Callable[
            [], Iterator[Union[SequenceSample, SentencePairSample, TokensSample, QASample, GenerationSample]]
        ],
        vocabulary: Vocabulary,
        transformer_model: str,
        tokens_per_batch: int,
        max_batch_size: Optional[int],
        section_size: int,
        prebatch: bool,
        materialize: bool,
        min_length: int,
        max_length: int,
        for_inference: bool,
        additional_special_tokens: Optional[List[str]] = None,
    ):
        self.sample_encoder = HFCustomGenerationSampleEncoder.from_transformer_model(
            transformer_model, additional_special_tokens
        )
        super().__init__(
            samples_iterator=samples_iterator,
            vocabulary=vocabulary,
            tokens_per_batch=tokens_per_batch,
            max_batch_size=max_batch_size,
            fields_batchers=self.sample_encoder.get_fields_batcher(),
            section_size=section_size,
            prebatch=prebatch,
            materialize=materialize,
            min_length=min_length,
            max_length=max_length if max_length != -1 else self.sample_encoder.get_tokenizer_max_length(),
            for_inference=for_inference,
            batching_fields=self.sample_encoder.get_batching_fields(for_inference),
        )

    def dataset_iterator_func(self) -> Iterable[Dict[str, Any]]:
        for generation_sample in self.samples_iterator():
            yield self.sample_encoder.sample2elem_dict(generation_sample, inference_mode=self.for_inference)

    def materialize_batches(self, dataset_elements: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        for group in self.sample_encoder.group_elements_on_materializations(
            dataset_elements, inference_mode=self.for_inference
        ):
            yield from super().materialize_batches(group)


class HFCustomGenerationSampleEncoder:
    @classmethod
    def from_transformer_model(cls, transformer_model: str, additional_special_tokens: Optional[List[str]]):
        if re.fullmatch("facebook/bart-(base|large)", transformer_model):
            return BartHFCustomGenerationSampleEncoder(transformer_model, additional_special_tokens)
        else:
            raise ValueError

    _shared_state = {}

    def __init__(self, transformer_model: str, additional_special_tokens: Optional[List[str]]):
        if "tokenizer" not in self._shared_state:
            self._shared_state["tokenizer"] = {}
        if transformer_model not in self._shared_state["tokenizer"]:
            self._shared_state["tokenizer"][transformer_model] = AutoTokenizer.from_pretrained(
                transformer_model,
                additional_special_tokens=list(additional_special_tokens)
                if additional_special_tokens is not None
                else None,
                use_fast=True,
                add_prefix_space=True,
            )
        self.tokenizer = self._shared_state["tokenizer"][transformer_model]

    def get_batching_fields(self, inference_mode: bool) -> List[str]:
        raise NotImplementedError

    def get_tokenizer_max_length(self) -> int:
        return self.tokenizer.model_max_length

    def get_fields_batcher(self) -> Dict[str, Callable]:
        raise NotImplementedError

    def sample2elem_dict(self, sample: GenerationSample, inference_mode: bool) -> Dict[str, Any]:
        raise NotImplementedError

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], inference_mode: bool
    ) -> List[List[Dict[str, Any]]]:
        return [dataset_elements]


class BartHFCustomGenerationSampleEncoder(EncDecHFGenerationSampleEncoder):
    def get_fields_batcher(self) -> Dict[str, Callable]:
        return {
            "input_ids": lambda lst: batchify(lst, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": lambda lst: batchify(lst, padding_value=0),
            "samples": None,
            "labels": lambda lst: batchify(lst, padding_value=-100),  # -100 == cross entropy ignore index
            "decoder_attention_mask": lambda lst: batchify(lst, padding_value=0),
            "decoder_start": lambda lst: batchify(
                lst, padding_value=-1
            ),  # -1 is to force the model to crash (padding should never happen)
        }

    def sample2elem_dict(self, sample: GenerationSample, inference_mode: bool) -> Dict[str, Any]:

        assert (
            sample.source_language is None and sample.target_language is None
        ), f"BartHFGenerationSampleEncoder does not support language specification"

        tokenization_output = self.tokenizer(sample.source_sequence, return_tensors="pt")
        elem_dict = {
            "input_ids": tokenization_output["input_ids"].squeeze(),
            "attention_mask": tokenization_output["attention_mask"].squeeze(),
        }

        if not inference_mode:
            if sample.target_sequence is not None:
                tokenization_output = self.tokenizer(sample.target_sequence, return_tensors="pt")
                input_ids = tokenization_output["input_ids"].squeeze()
                decoder_attention_mask = tokenization_output["attention_mask"].squeeze()

                compositional_tokens = getattr(sample, "compositional_tokens", None)
                if compositional_tokens is not None:
                    compositional_tokens = [f"<{ct}>" for ct in compositional_tokens]
                    compositional_token_ids = [self.tokenizer.convert_tokens_to_ids(ct) for ct in compositional_tokens]
                    input_ids = torch.cat([torch.tensor(compositional_token_ids, dtype=input_ids.dtype), input_ids[1:]], dim=0)
                    if len(compositional_tokens) > 1:
                        decoder_attention_mask = torch.cat(
                            [torch.ones(len(compositional_tokens) - 1, dtype=torch.int), decoder_attention_mask],
                            dim=0
                        )
                elem_dict.update(
                    **{
                        "labels": input_ids,
                        "decoder_attention_mask": decoder_attention_mask,
                    }
                )
        else:
            elem_dict["decoder_start"] = torch.tensor([self.tokenizer.eos_token_id])

        elem_dict["samples"] = sample
        return elem_dict

    def group_elements_on_materializations(
        self, dataset_elements: List[Dict[str, Any]], inference_mode: bool
    ) -> List[List[Dict[str, Any]]]:
        if not inference_mode:
            return [dataset_elements]

        groups = collections.defaultdict(list)

        for de in dataset_elements:
            groups[de["samples"].dataset_id].append(de)

        return [group for group_len, group in groups.items()]
