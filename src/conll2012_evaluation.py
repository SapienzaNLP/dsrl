import json
import logging
from pprint import pprint
import re
import subprocess
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm

import torch
from classy.data.data_drivers import GenerationSample
from classy.evaluation.base import Evaluation
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class Conll2012Evaluation(Evaluation):

    def __init__(
            self,
            frames_path: str,
            argm_definitions_path: str,
            gold_input_path: str,
            gold_output_path: str,
            pred_output_path: str,
            scorer_path: str,
            language_model_name: str,
    ) -> None:
        super().__init__()

        with open(frames_path) as f:
            self.frames = json.load(f)

        with open(argm_definitions_path) as f:
            self.argm_definitions = json.load(f)

        self.gold_input_path = gold_input_path
        self.gold_output_path = gold_output_path
        self.pred_output_path = pred_output_path
        self.scorer_path = scorer_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModel.from_pretrained(language_model_name)
        self.language_model.to(self.device)
        self.language_model.eval()

        print("Building conll2012 sense definitions index")
        self.sense_index = self.build_sense_definitions_index()

        print("Building conll2012 role definitions index")
        self.role_index = self.build_role_definitions_index()

    def __call__(self, predicted_samples: List[Tuple[GenerationSample, str]]) -> Dict:

        # source target example:
        # A record date has n't been <p> set </p> .	set : establish <role> date : thing set <role> n't : negation
        # each predicted item is separated by the special token <role>

        predictions = {}

        for sample, predicted_sequence in predicted_samples:
            sentence_index = sample.id
            predicate_index = sample.predicate_indices[0]
            predicate_type = sample.predicate_type

            source_sequence = (
                sample.source_sequence
                    .replace('<p>', '').replace('</p>', '')
                    .replace('<propbank>', '').replace('<framenet>', '')
                    .replace('<span-srl>', '').replace('<dep-srl>', '')
                    .strip()
            )
            source_tokens = source_sequence.split()
            
            predicted_sequence = (
                re.sub(r'</?s>', '', predicted_sequence)
                    .replace('<pad>', '')
                    .replace('<propbank>', '').replace('<framenet>', '')
                    .replace('<dep-srl>', '').replace('<span-srl>', '')
                    .strip()
            )
            sense_definition, *predicted_sentence = predicted_sequence.split('.')
            sense_definition = sense_definition.strip()
            predicted_sentence = '.'.join(predicted_sentence).strip()

            sense_label = self.choose_sense_label(sense_definition, predicate_type)
            role_labels = self.choose_role_labels(source_tokens, predicted_sentence, sense_label)

            if sentence_index not in predictions:
                predictions[sentence_index] = {}

            predicate_labels = {
                'sense': sense_label.replace(f'.{predicate_type}.', '.'),
                'roles': role_labels,
            }
            predictions[sentence_index][predicate_index] = predicate_labels

        self.write_as_conll(predictions)
        metrics = self.run_scorer()
        return metrics

    def choose_sense_label(self, sense_definition: str, predicate_type: str) -> str:
        sense_definition = sense_definition.strip()

        if ':' not in sense_definition:
            return 'unk.01'

        predicate_lemma = sense_definition.split(':')[0].strip().lower()
        if predicate_lemma not in self.frames and ' ' in predicate_lemma:
            predicate_lemma = predicate_lemma.split()[0]
        if predicate_lemma not in self.frames and predicate_lemma.endswith('ed'):
            predicate_lemma = predicate_lemma[:-2]
        if predicate_lemma not in self.frames and predicate_lemma.endswith('ing'):
            predicate_lemma = predicate_lemma[:-3]
        if predicate_lemma not in self.frames and predicate_lemma + 'e' in self.frames:
            predicate_lemma = predicate_lemma + 'e'
        if predicate_lemma not in self.frames and predicate_lemma.endswith('s'):
            predicate_lemma = predicate_lemma[:-1]
        if predicate_lemma not in self.frames:
            return 'unk.01'

        if sense_definition in self.sense_index:
            sense_embedding = self.sense_index[sense_definition]
        else:
            sense_embedding = self.compute_embedding(sense_definition)

        predicate_frames = self.frames[predicate_lemma]
        candidate_senses = sorted(list(k for k in predicate_frames.keys() if f'.{predicate_type}.' in k))

        sense, max_similarity = None, None
        for candidate_sense in candidate_senses:
            candidate_definition = predicate_frames[candidate_sense]['definition']
            if sense_definition == candidate_definition:
                sense, max_similarity = candidate_sense, 1.0
                break

        if sense is None:
            for candidate_sense in candidate_senses:
                candidate_definition = predicate_frames[candidate_sense]['definition']
                candidate_embedding = self.sense_index[candidate_definition]
                similarity = F.cosine_similarity(sense_embedding, candidate_embedding, dim=0).item()
                if sense is None or similarity > max_similarity:
                    sense, max_similarity = candidate_sense, similarity

        if sense is None:
            sense = 'none.01'

        return sense

    def choose_role_labels(self, source_tokens: List[str], predicted_sentence: str, predicate_sense: str) -> Dict[int, str]:
        predicate_lemma = predicate_sense.split('.')[0]
        core_roleset = self.frames[predicate_lemma][predicate_sense][
            'roleset'] if predicate_lemma in self.frames else {'ARG0': 'agent', 'ARG1': 'theme'}
        full_roleset = core_roleset.copy()

        for argm, argm_definition in self.argm_definitions.items():
            if argm not in full_roleset:
                full_roleset[argm] = argm_definition

        predicted_tokens = predicted_sentence.split()
        offset = 0
        argument = None
        argument_start_index, argument_end_index = 0, 0
        inside_role_definition = False
        current_role_definition = []
        role_definitions = {}

        for token_index, token in enumerate(predicted_tokens):
            if token == '[':
                argument_start_index = token_index + 1
                continue
            if token == ']{':
                argument_end_index = token_index
                inside_role_definition = True
                continue
            if token == '}':
                inside_role_definition = False
                argument = predicted_tokens[argument_start_index:argument_end_index]
                asi = argument_start_index - offset
                aei = argument_end_index - offset
                role_definitions[(asi, aei)] = (argument, current_role_definition)
                offset += 3 + len(current_role_definition)
                current_role_definition = []
                continue

            if inside_role_definition:
                current_role_definition.append(token)
        
        roles = []
        for (argument_start_index, argument_end_index), (argument, role_definition) in role_definitions.items():
            if not role_definition:
                continue
            elif role_definition[0] == '<reference-to>':
                role_prefix = 'R-'
                role_definition = role_definition[1:]
            elif role_definition[0] == '<continuation-of>':
                role_prefix = 'C-'
                role_definition = role_definition[1:]
            else:
                role_prefix = ''
            
            role_definition = ' '.join(role_definition)
            role_definition = role_definition.strip()
            if not argument or argument[0] not in source_tokens:
                continue

            if role_definition in self.role_index:
                role_embedding = self.role_index[role_definition]
            else:
                role_embedding = self.compute_embedding(role_definition)

            candidate_roles = sorted(list(full_roleset.keys()))
            role, max_similarity, core_bonus = None, 0.5, 0.1
            for candidate_role in candidate_roles:
                candidate_definition = full_roleset[candidate_role]
                if role_definition == candidate_definition:
                    role, similarity = candidate_role, 1.0

                    if candidate_role in core_roleset:
                        similarity += core_bonus
                    if similarity > max_similarity:
                        role, max_similarity = candidate_role, similarity

            if role is None:
                for candidate_role in candidate_roles:
                    candidate_definition = full_roleset[candidate_role]
                    candidate_embedding = self.role_index[candidate_definition]
                    similarity = F.cosine_similarity(role_embedding, candidate_embedding, dim=0).item()

                    if candidate_role in core_roleset:
                        similarity += core_bonus
                    if similarity > max_similarity:
                        role, max_similarity = candidate_role, similarity

            if role is None:
                continue

            occurrences = []
            argument_length = len(argument)
            sentence_length = len(source_tokens)
            for span_start in range(len(source_tokens)):
                span_end = span_start + argument_length
                if source_tokens[span_start] != argument[0]:
                    continue
                if span_end >= sentence_length or source_tokens[span_end - 1] != argument[-1]:
                    continue
                occurrences.append((span_start, span_end))

            closest_occurrence_index, minimum_distance = None, None
            for span_start, span_end in occurrences:
                distance = abs(span_start - argument_start_index) + abs(span_end - argument_end_index)
                if closest_occurrence_index is None or distance <= minimum_distance:
                    closest_occurrence_index = (span_start, span_end)
                    minimum_distance = distance

            if closest_occurrence_index is not None:
                roles.append({
                    'role': f'{role_prefix}{role}',
                    'start': closest_occurrence_index[0],
                    'end': closest_occurrence_index[1]
                })

        return roles

    def write_as_conll(self, predictions: Dict) -> None:

        with open(self.gold_input_path) as f_in, open(self.gold_output_path, 'w') as g_out, open(self.pred_output_path, 'w') as p_out:
            sentence_id: int = 0
            gold_predicate_indices: List[int] = []
            gold_predicates: List[str] = []
            gold_roles: List[List[str]] = []

            for line in f_in:
                if line[0] == '#':
                    continue

                line = line.strip()

                # End of sentence or end of file.
                if not line:
                    sentence_length = len(gold_predicates)
                    pred_roles = []
                    for predicate_index in gold_predicate_indices:
                        null_roles = ['*'] * sentence_length
                        null_roles[predicate_index] = '(V*)'
                        pred_roles.append(null_roles)

                    if sentence_id in predictions:
                        sentence_predictions = predictions[sentence_id]

                        for i, predicate_index in enumerate(gold_predicate_indices):
                            if predicate_index not in sentence_predictions:
                                continue

                            is_argument = [False] * len(gold_predicates)
                            is_argument[predicate_index] = True

                            predicate_predictions = sentence_predictions[predicate_index]
                            for role_info in predicate_predictions['roles']:
                                role, start, end = role_info['role'], role_info['start'], role_info['end']
                                if True in is_argument[start:end]:
                                    continue
                                is_argument[start:end] = [True] * (end - start)
                                if start == end - 1:
                                    pred_roles[i][start] = f'({role}*)'
                                else:
                                    pred_roles[i][start] = f'({role}*'
                                    pred_roles[i][end - 1] = f'*)'

                    pred_roles = list(map(list, zip(*pred_roles)))

                    for i in range(len(gold_predicates)):
                        gold_output_line = gold_predicates[i]
                        pred_output_line = gold_predicates[i]
                        if gold_roles:
                            gold_output_line = gold_output_line + '\t' + '\t'.join(gold_roles[i])
                        if pred_roles:
                            pred_output_line = pred_output_line + '\t' + '\t'.join(pred_roles[i])

                        g_out.write(f'{gold_output_line}\n')
                        p_out.write(f'{pred_output_line}\n')
                    g_out.write('\n')
                    p_out.write('\n')

                    sentence_id += 1
                    gold_predicate_indices: List[int] = []
                    gold_predicates: List[str] = []
                    gold_roles: List[List[str]] = []
                    continue

                line_parts = line.split()
                if line_parts[7] != '-':
                    gold_predicates.append(line_parts[6])
                    gold_predicate_indices.append(len(gold_predicates) - 1)
                else:
                    gold_predicates.append('-')
                gold_roles.append(line_parts[11:-1])

    def run_scorer(self):
        arguments = ['perl', self.scorer_path, self.gold_output_path, self.pred_output_path]
        scorer_output = subprocess.run(arguments, capture_output=True, text=True).stdout
        print(scorer_output)

        precision, recall, f1 = None, None, None
        for line in scorer_output.splitlines():
            if 'Overall' not in line:
                continue

            line_parts = line.split()
            precision, recall, f1 = line_parts[-3], line_parts[-2], line_parts[-1]
            precision = float(precision)
            recall = float(recall)
            f1 = float(f1)
            break

        assert all([m is not None for m in [precision, recall, f1]])

        return {
            'conll2012_arg_precision': precision,
            'conll2012_arg_recall': recall,
            'conll2012_arg_f1': f1,
        }

    def build_sense_definitions_index(self) -> Dict[str, torch.Tensor]:
        index: Dict[str, torch.Tensor] = {}

        for predicate_lemma, predicate_frames in tqdm(self.frames.items()):

            for predicate_sense, predicate_sense_info in predicate_frames.items():
                sense_definition = predicate_sense_info['definition']
                sense_definition = sense_definition.strip()
                if sense_definition in index:
                    continue

                sense_embedding = self.compute_embedding(sense_definition)
                index[sense_definition] = sense_embedding

        return index

    def build_role_definitions_index(self) -> Dict[str, torch.Tensor]:
        index: Dict[str, torch.Tensor] = {}

        for predicate_lemma, predicate_frames in tqdm(self.frames.items()):

            for predicate_sense, predicate_sense_info in predicate_frames.items():
                roleset = predicate_sense_info['roleset']

                for role, role_definition in roleset.items():
                    role_definition = role_definition.strip()
                    if role_definition in index:
                        continue

                    role_embedding = self.compute_embedding(role_definition)
                    index[role_definition] = role_embedding

        for role, role_definition in self.argm_definitions.items():
            if role_definition in index:
                continue

            role_embedding = self.compute_embedding(role_definition)
            index[role_definition] = role_embedding

        return index

    def compute_embedding(self, text: str, strategy: Optional[str] = 'pooling', limit: int = 256) -> torch.Tensor:
        model_inputs = self.tokenizer(text[:limit], return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_outputs = self.language_model(**model_inputs)

        if strategy == 'pooling':
            return torch.squeeze(model_outputs.pooler_output).to('cpu')
        elif strategy == 'average':
            return torch.squeeze(torch.mean(model_outputs.last_hidden_state, dim=1)).to('cpu')

        raise NotImplementedError()
