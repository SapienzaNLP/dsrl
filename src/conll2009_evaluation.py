import json
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


class Conll2009Evaluation(Evaluation):

    def __init__(
            self,
            frames_path: str,
            argm_definitions_path: str,
            conll_path: str,
            output_path: str,
            scorer_path: str,
            language_model_name: str,
    ) -> None:
        super().__init__()

        with open(frames_path) as f:
            self.frames = json.load(f)

        with open(argm_definitions_path) as f:
            self.argm_definitions = json.load(f)

        self.conll_path = conll_path
        self.output_path = output_path
        self.scorer_path = scorer_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModel.from_pretrained(language_model_name)
        self.language_model.to(self.device)
        self.language_model.eval()

        print("Building conll2009 sense definitions index")
        self.sense_index = self.build_sense_definitions_index()

        print("Building conll2009 role definitions index")
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
            'roleset'] if predicate_lemma in self.frames else {'A0': 'agent', 'A1': 'theme'}
        full_roleset = core_roleset.copy()

        for argm, argm_definition in self.argm_definitions.items():
            if argm not in full_roleset:
                full_roleset[argm] = argm_definition

        predicted_tokens = predicted_sentence.split(' ')
        offset = 0
        argument = None
        argument_index = 0
        inside_role_definition = False
        current_role_definition = []
        role_definitions = {}

        for token_index, token in enumerate(predicted_tokens):
            if token == '{':
                inside_role_definition = True
                argument_index = token_index - 1
                argument = predicted_tokens[argument_index]
                continue
            if token == '}':
                inside_role_definition = False
                role_definitions[argument_index - offset] = (argument, current_role_definition)
                offset += 2 + len(current_role_definition)
                current_role_definition = []
                continue

            if inside_role_definition:
                current_role_definition.append(token)
        
        roles = {}
        for argument_index, (argument, role_definition) in role_definitions.items():
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
            if argument not in source_tokens:
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

            closest_occurrence_index, minimum_distance = None, None
            for occurrence_index, occurrence in enumerate(source_tokens):
                if occurrence != argument:
                    continue
                distance = abs(occurrence_index - argument_index)
                if closest_occurrence_index is None or distance <= minimum_distance:
                    closest_occurrence_index = occurrence_index
                    minimum_distance = distance

            if closest_occurrence_index is not None:
                argument_index = closest_occurrence_index
                roles[argument_index] = f'{role_prefix}{role}'
        
        return roles

    def write_as_conll(self, predictions: Dict) -> None:

        with open(self.conll_path) as f_in, open(self.output_path, 'w') as f_out:
            sentence_id: int = 0
            sentence_parts: List[List[str]] = []

            for line in f_in:
                line = line.strip()

                # End of sentence or end of file.
                if not line:
                    sentence_predicates = ['_'] * len(sentence_parts)
                    sentence_senses = ['_'] * len(sentence_parts)
                    sentence_roles = []

                    if sentence_id in predictions:
                        sentence_predictions = predictions[sentence_id]
                        predicate_indices = sorted(list(sentence_predictions.keys()))

                        for predicate_index in predicate_indices:
                            predicate_predictions = sentence_predictions[predicate_index]
                            sentence_predicates[predicate_index] = 'Y'
                            sentence_senses[predicate_index] = predicate_predictions['sense']
                            predicate_roles = ['_'] * len(sentence_parts)

                            for role_index, role in predicate_predictions['roles'].items():
                                if role_index < len(predicate_roles):
                                    predicate_roles[role_index] = role
                                else:
                                    print(f'Role index {role_index} for predicate index {predicate_index} not found in sentence {sentence_id}.')

                            sentence_roles.append(predicate_roles)

                    sentence_roles = list(map(list, zip(*sentence_roles)))
                    for i in range(len(sentence_parts)):
                        sentence_parts[i].extend([sentence_predicates[i], sentence_senses[i]])
                        if sentence_roles:
                            sentence_parts[i].extend(sentence_roles[i])

                    for i in range(len(sentence_parts)):
                        output_line = '\t'.join(sentence_parts[i])
                        f_out.write(f'{output_line}\n')
                    f_out.write('\n')

                    sentence_id += 1
                    sentence_parts: List[List[str]] = []
                    continue

                line_parts = line.split()[:12]
                sentence_parts.append(line_parts)

    def run_scorer(self):
        arguments = ['perl', self.scorer_path, '-q', '-g', self.conll_path, '-s', self.output_path]
        scorer_output = subprocess.run(arguments, capture_output=True, text=True).stdout
        print(scorer_output)
        precision, recall, f1, predicate_f1, arg_precision, arg_recall, arg_f1 = \
            0., 0., 0., 0., 0., 0., 0.

        for line in scorer_output.splitlines():
            if 'Labeled F1:' in line:
                f1 = float(line.split(':')[-1].strip())
            elif 'Labeled precision:' in line:
                arg_precision, predicate_f1 = self.extract_scores_from_formula(line, predicate_f1)
                precision = float(line.split('=')[-1].replace('%', '').strip())
            elif 'Labeled recall:' in line:
                arg_recall, predicate_f1 = self.extract_scores_from_formula(line, predicate_f1)
                recall = float(line.split('=')[-1].replace('%', '').strip())

        if arg_precision + arg_recall != 0.0:
            arg_f1 = 2 * arg_precision * arg_recall / (arg_precision + arg_recall)

        return {
            'conll2009_precision': precision,
            'conll2009_recall': recall,
            'conll2009_f1': f1,
            'conll2009_predicate_f1': predicate_f1,
            'conll2009_arg_precision': arg_precision,
            'conll2009_arg_recall': arg_recall,
            'conll2009_arg_f1': arg_f1
        }

    def extract_scores_from_formula(self, line: str, predicate_f1: float):
        groups = re.split("[()]", line.strip())
        predicate_sense_num = float(groups[1].split("+")[1].strip())
        predicate_sense_den = float(groups[3].split("+")[1].strip())
        if predicate_sense_den != 0.0:
            predicate_f1 = (predicate_sense_num / predicate_sense_den) * 100
        arg_num = float(groups[1].split("+")[0].strip())
        arg_den = float(groups[3].split("+")[0].strip())
        arg_score = 0.0
        if arg_den != 0.0:
            arg_score = (arg_num / arg_den) * 100
        return arg_score, predicate_f1

    def build_sense_definitions_index(self) -> Dict[str, torch.Tensor]:
        index: Dict[str, torch.Tensor] = {}

        for predicate_lemma, predicate_frames in tqdm(self.frames.items()):

            for predicate_sense, predicate_sense_info in predicate_frames.items():
                sense_definition = predicate_sense_info['definition']
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
