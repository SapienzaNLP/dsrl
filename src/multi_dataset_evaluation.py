from typing import List, Tuple, Union, Dict

from classy.data.data_drivers import SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample
from classy.evaluation.base import Evaluation


class MultiDatasetEvaluation(Evaluation):

    def __init__(self, evaluations: Dict[str, Evaluation]):
        self.evaluations = evaluations

    def __call__(self, predicted_samples: List[
        Tuple[
            Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample],
            Union[str, List[str], Tuple[int, int]],
        ]
    ]) -> Dict:
        final_dict = dict()
        for eval_name, eval_instance in self.evaluations.items():
            eval_output = eval_instance(predicted_samples)
            for key, value in eval_output.items():
                final_dict[f'{key}'] = value
        f1s = [final_dict[f'{key}'] for key in final_dict.keys() if 'arg_f1' in key]
        final_dict['overall_f1'] = sum(f1s) / len(f1s)
        return final_dict


class DatasetSpecificEvaluation(Evaluation):

    def __init__(self, dataset_id: str, evaluation: Evaluation):
        self.dataset_id = dataset_id
        self.evaluation = evaluation

    def __call__(self, predicted_samples: List[
        Tuple[
            Union[SentencePairSample, SequenceSample, TokensSample, QASample, GenerationSample],
            Union[str, List[str], Tuple[int, int]],
        ]
    ]) -> Dict:
        return self.evaluation([ps for ps in predicted_samples if ps[0].dataset_id == self.dataset_id])
