import json
import logging
import re
import subprocess
from typing import List, Set, Tuple, Dict, Optional

import Levenshtein as lev
from tqdm import tqdm

import torch
from classy.data.data_drivers import GenerationSample
from classy.evaluation.base import Evaluation
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class FramenetEvaluation(Evaluation):

    def __init__(
            self,
            frames_path: str,
            fr_frames_path: str,
            fr_relations_path: str,
            gold_input_path: str,
            gold_output_path: str,
            pred_output_path: str,
            scorer_path: str,
            language_model_name: str,
    ) -> None:
        super().__init__()

        with open(frames_path) as f:
            self.frames = json.load(f)

        self.use_lexical_unit = True
        self.use_gold_frame = False
        self.frames_path = fr_frames_path
        self.relations_path = fr_relations_path
        self.gold_input_path = gold_input_path
        self.gold_output_path = gold_output_path
        self.pred_output_path = pred_output_path
        self.scorer_path = scorer_path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        self.language_model = AutoModel.from_pretrained(language_model_name)
        self.language_model.to(self.device)
        self.language_model.eval()

        print("Building framenet17 sense definitions index")
        self.sense_index = self.build_sense_definitions_index()

        print("Building framenet17 role definitions index")
        self.role_index = self.build_role_definitions_index()

    def __call__(self, predicted_samples: List[Tuple[GenerationSample, str]]) -> Dict:

        # source target example:
        # A record date has n't been <p> set </p> .	set : establish <role> date : thing set <role> n't : negation
        # each predicted item is separated by the special token <role>

        predictions = {}

        for sample, predicted_sequence in predicted_samples:
            sentence_index = sample.id
            predicate_indices = sample.predicate_indices
            predicate_sense = sample.predicate_sense
            lexical_unit = sample.lexical_unit
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

            predicate, sense_label = self.choose_sense_label(sense_definition, lexical_unit, predicate_type)
            if self.use_gold_frame:
                role_labels = self.choose_role_labels(source_tokens, predicted_sentence, predicate_sense, predicate)
            else:
                role_labels = self.choose_role_labels(source_tokens, predicted_sentence, sense_label, predicate)

            if sentence_index not in predictions:
                predictions[sentence_index] = {}

            predicate_labels = {
                'tokens': source_tokens,
                'sense': sense_label,
                'roles': role_labels,
            }
            predictions[sentence_index][predicate_indices] = predicate_labels

        gold = self.read_from_file(self.gold_input_path)
        self.write_as_xml(gold, self.gold_output_path)
        self.write_as_xml(predictions, self.pred_output_path)

        # metrics = self.compute_metrics(predictions)
        metrics = self.run_scorer()
        sense_metrics = self.run_scorer(frames_only=True)
        arg_precision = 100. * (metrics['precision_num'] - sense_metrics['precision_num']) / (metrics['precision_den'] - sense_metrics['precision_den']) if metrics['precision_den'] - sense_metrics['precision_den'] != 0 else 0.
        arg_recall = 100 * (metrics['recall_num'] - sense_metrics['recall_num']) / (metrics['recall_den'] - sense_metrics['recall_den']) if metrics['recall_den'] - sense_metrics['recall_den'] != 0 else 0.
        arg_f1 = 2 * arg_precision * arg_recall / (arg_precision + arg_recall) if arg_precision + arg_recall != 0 else 0.
        metrics = {
            'framenet_arg_precision': arg_precision,
            'framenet_arg_recall': arg_recall,
            'framenet_arg_f1': arg_f1,
            'framenet_predicate_f1': sense_metrics['f1'],
            'framenet_precision': metrics['precision'],
            'framenet_recall': metrics['recall'],
            'framenet_f1': metrics['f1'],
        }
        
        return metrics

    def choose_sense_label(self, sense_definition: str, lexical_unit: str, predicate_type: str) -> str:
        sense_definition = sense_definition.strip()

        if self.use_lexical_unit:
            predicate = lexical_unit
        else:
            predicate = self.get_predicate_label(sense_definition, predicate_type)
        
        if predicate == 'unknown' or predicate not in self.frames:
            return 'unknown', 'Unknown'

        if sense_definition in self.sense_index:
            sense_embedding = self.sense_index[sense_definition]
        else:
            sense_embedding = self.compute_embedding(sense_definition)

        predicate_frames = self.frames[predicate]
        candidate_senses = sorted(list(predicate_frames.keys()))

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
            sense = 'None'

        return predicate, sense

    def get_predicate_label(self, sense_definition: str, predicate_type: str = None) -> Tuple[str, str]:
        if ':' not in predicate:
            return 'unknown'
        
        predicate_lemma = sense_definition.split(':')[0].strip()
        predicate = f'{predicate_lemma}.{predicate_type}'
        if predicate in self.frames:
            return predicate
        if predicate.lower() in self.frames:
            return predicate.lower()
        
        predicate = predicate.lower()

        if ' ' in predicate_lemma:
            predicate_lemma = predicate_lemma.split()[0]
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        if predicate_lemma.endswith('ed') and f'{predicate_lemma[:-2]}.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma[:-2]
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        elif f'{predicate_lemma}ed.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma + 'ed'
            predicate = f'{predicate_lemma}.{predicate_type}'

        elif predicate_lemma.endswith('er') and f'{predicate_lemma[:-2]}.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma[:-2]
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        elif predicate_lemma.endswith('ing') and f'{predicate_lemma[:-3]}.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma[:-3]
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        elif predicate_lemma.endswith('ied') and f'{predicate_lemma[:-3]}y.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma[:-3] + 'y'
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        elif f'{predicate_lemma}e.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma + 'e'
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        elif predicate_lemma.endswith('s') and f'{predicate_lemma[:-1]}.{predicate_type}' in self.frames:
            predicate_lemma = predicate_lemma[:-1]
            predicate = f'{predicate_lemma}.{predicate_type}'
        
        else:
            found = False
            for frame in self.frames:
                frame_lemma = frame.split('.')[0]
                if ' ' in frame and predicate_lemma in frame:
                    found = True
                    predicate = f'{frame}.{predicate_type}'
                    break
                if lev.distance(predicate_lemma, frame, score_cutoff=1) <= 1:
                    found = True
                    predicate = f'{frame}.{predicate_type}'
                    break
            
            if not found:
                predicate = 'unknown'
        
        return predicate

    def choose_role_labels(self, source_tokens: List[str], predicted_sentence: str, predicate_sense: str, predicate: str) -> Dict[int, str]:
        if predicate_sense == 'Unknown' or predicate_sense == 'None':
            return []

        core_roleset = self.frames[predicate][predicate_sense]['roleset']['core']
        full_roleset = core_roleset.copy()
        full_roleset.update(self.frames[predicate][predicate_sense]['roleset']['non-core'])

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

            role_definition = ' '.join(role_definition)
            role_definition = role_definition.strip()
            if not argument or argument[0] not in source_tokens:
                continue

            if role_definition in self.role_index:
                role_embedding = self.role_index[role_definition]
            else:
                role_embedding = self.compute_embedding(role_definition)
            
            candidate_roles = sorted(list(full_roleset.keys()))
            role, max_similarity, core_bonus = None, 0.5, 0.0
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
                if span_end > sentence_length or source_tokens[span_end - 1] != argument[-1]:
                    continue
                occurrences.append((span_start, span_end))

            closest_occurrence_index, minimum_distance = None, None
            for span_start, span_end in occurrences:
                distance = min(abs(span_start - argument_start_index), abs(span_end - argument_start_index))
                if closest_occurrence_index is None or distance <= minimum_distance:
                    closest_occurrence_index = (span_start, span_end)
                    minimum_distance = distance

            if closest_occurrence_index is not None:
                roles.append({
                    'role': role,
                    'start': closest_occurrence_index[0],
                    'end': closest_occurrence_index[1]
                })

        return roles

    def compute_metrics(self, predictions: Dict):
        role_tp, role_fp, role_fn = 0., 0., 0.
        frame_correct, frame_total = 0., 0.

        with open(self.gold_input_path) as f_in:
            sentence_id: int = 0
            gold_predicate_indices: List[int] = []
            gold_predicate: str = None
            gold_predicate_sense: str = None
            gold_current_role: str = None
            gold_current_role_start: int = None
            gold_predicate_roles: Set[Tuple[str, int, int]] = set()

            for line in f_in:
                if line[0] == '#':
                    continue

                line = line.strip()

                # End of sentence or end of file.
                if not line:
                    predicate_indices = tuple(p for p in gold_predicate_indices)
                    sentence_length = len(gold_predicate_roles)

                    frame_total += len(gold_predicate_indices)

                    if gold_current_role is not None:
                        gold_predicate_roles.add((gold_current_role, gold_current_role_start, token_id + 1))

                    if sentence_id in predictions:
                        sentence_predictions = predictions[sentence_id]
                        if predicate_indices not in sentence_predictions:
                            continue

                        predicate_predictions = sentence_predictions[predicate_indices]
                        if predicate_predictions['sense'] == gold_predicate_sense:
                            frame_correct += len(gold_predicate_indices)

                        is_argument = [False] * sentence_length
                        for role_info in predicate_predictions['roles']:
                            p_role, p_start, p_end = role_info['role'], role_info['start'], role_info['end']
                            if True in is_argument[p_start:p_end]:
                                continue
                            is_argument[p_start:p_end] = [True] * (p_end - p_start)

                            if any(p_index >= p_start and p_index < p_end for p_index in gold_predicate_indices):
                                role_score = 0.5
                            elif p_role in self.frames[gold_predicate][gold_predicate_sense]['roleset']['core']:
                                role_score = 1.
                            else:
                                role_score = 0.5
                            
                            if (p_role, p_start, p_end) in gold_predicate_roles:
                                role_tp += role_score
                            else:
                                role_fp += role_score
                        
                        for g_role, g_start, g_end in gold_predicate_roles:
                            correct = False
                            for role_info in predicate_predictions['roles']:
                                p_role, p_start, p_end = role_info['role'], role_info['start'], role_info['end']
                                if g_role == p_role and g_start == p_start and g_end == p_end:
                                    correct = True
                            
                            if any(g_index >= g_start and g_index < g_end for g_index in gold_predicate_indices):
                                role_score = 0.5
                            elif g_role in self.frames[gold_predicate][gold_predicate_sense]['roleset']['core']:
                                role_score = 1.
                            else:
                                role_score = 0.5
                            
                            if not correct:
                                role_fn += role_score

                    sentence_id += 1
                    gold_predicate_indices: List[int] = []
                    gold_predicate: str = None
                    gold_predicate_sense: str = None
                    gold_current_role: str = None
                    gold_predicate_roles: Set[Tuple[str, int, int]] = set()
                    continue

                line_parts = line.split('\t')
                token_id, token, _, predicted_lemma, _, predicted_pos, _, _, _, _, _, _, predicate, predicate_sense, role = line_parts
                token_id = int(token_id) - 1
                role_type, role = role[0], role[2:]

                if gold_current_role is not None and (role_type == 'B' or role_type == 'S' or role_type == 'O'):
                    gold_predicate_roles.add((gold_current_role, gold_current_role_start, token_id))
                    gold_current_role = None
                
                if role_type == 'S':
                    gold_predicate_roles.add((role, token_id, token_id + 1))
                    gold_current_role = None
                elif role_type == 'B':
                    gold_current_role = role
                    gold_current_role_start = token_id
                
                is_predicate = predicate_sense != '_'
                if is_predicate:
                    gold_predicate = predicate
                    gold_predicate_sense = predicate_sense
                    gold_predicate_indices.append(token_id)

        predicate_f1 = frame_correct / frame_total

        arg_precision = role_tp / (role_tp + role_fp) if role_tp + role_fp != 0. else 0.
        arg_recall = role_tp / (role_tp + role_fn) if role_tp + role_fn != 0. else 0.
        arg_f1 = 2 * (arg_precision * arg_recall) / (arg_precision + arg_recall) if arg_precision + arg_recall != 0. else 0.

        precision = (role_tp + frame_correct) / (role_tp + role_fp + frame_total) if arg_precision != 0. else 0.
        recall = (role_tp + frame_correct) / (role_tp + role_fn + frame_total) if arg_recall != 0. else 0.
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0. else 0.

        metrics = {
            'predicate_f1': 100. * predicate_f1,
            'arg_precision': 100. * arg_precision,
            'arg_recall': 100. * arg_recall,
            'arg_f1': 100. * arg_f1,
            'precision': 100. * precision,
            'recall': 100. * recall,
            'f1': 100. * f1,
        }

        print(metrics)
        return metrics

    def write_as_xml(self, predictions: Dict, out_path: str) -> None:
        """
        Write predictions as an XML file.

        :param predictions: Predictions.
        """
        with open(out_path, 'w') as f_out:
            f_out.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f_out.write('<corpus name="name">\n')
            f_out.write('<documents>\n')
            f_out.write('<document ID="1" description="description">\n')
            f_out.write('<paragraphs>\n')
            f_out.write('<paragraph ID="1" document_order="1">\n')
            f_out.write('<sentences>\n')

            sentence_ids = sorted(list(predictions.keys()))

            for sentence_id in sentence_ids:
                sentence_predictions = predictions[sentence_id]

                for i, (predicate_indices, predicate_predictions) in enumerate(sentence_predictions.items()):
                    tokens = predicate_predictions['tokens']
                    t2c = self.token2char(tokens)
                    text = ' '.join(tokens)
                    predicate_sense = predicate_predictions['sense']
                    f_out.write(f'\t<sentence ID="{sentence_id}">\n')
                    f_out.write(f'\t\t<text>{text}</text>\n')
                    f_out.write('\t\t<annotationSets>\n')
                    f_out.write(f'\t\t\t<annotationSet ID="{sentence_id}00{i}00" frameName="{predicate_sense}">\n')
                    f_out.write('\t\t\t\t<layers>\n')
                    f_out.write(f'\t\t\t\t\t<layer ID="{sentence_id}00{i}0001" name="Target">\n')
                    f_out.write('\t\t\t\t\t\t<labels>\n')

                    for j, predicate_index in enumerate(predicate_indices, start=1):
                        start_index, end_index = t2c[predicate_index]
                        f_out.write(f'\t\t\t\t\t\t\t<label ID="{sentence_id}00{i}00010{j}" name="Target" start="{start_index}" end="{end_index}"/>\n')

                    f_out.write('\t\t\t\t\t\t</labels>\n')
                    f_out.write('\t\t\t\t\t</layer>\n')
                    f_out.write(f'\t\t\t\t\t<layer ID="{sentence_id}00{i}0001" name="FE">\n')
                    f_out.write('\t\t\t\t\t\t<labels>\n')

                    for j, role_info in enumerate(predicate_predictions['roles'], start=1):
                        role_name, start_index, end_index = role_info['role'], role_info['start'], role_info['end']
                        start_index = t2c[start_index][0]
                        end_index = t2c[end_index - 1][1]
                        f_out.write(f'\t\t\t\t\t\t\t<label ID="{sentence_id}00{i}00020{j}" name="{role_name}" start="{start_index}" end="{end_index}"/>\n')

                    f_out.write('\t\t\t\t\t\t</labels>\n')
                    f_out.write('\t\t\t\t\t</layer>\n')
                    f_out.write('\t\t\t\t</layers>\n')
                    f_out.write('\t\t\t</annotationSet>\n')
                    f_out.write('\t\t</annotationSets>\n')
                    f_out.write('\t</sentence>\n')

            f_out.write('</sentences>\n')
            f_out.write('</paragraph>\n')
            f_out.write('</paragraphs>\n')
            f_out.write('</document>\n')
            f_out.write('</documents>\n')
            f_out.write('</corpus>\n')

    def token2char(self, tokens: List[str]) -> Dict[str, int]:
        """
        Convert tokens to character offsets.

        :param tokens: Tokens.
        :return: Character offsets.
        """
        char_offsets = {}
        offset = 0
        for token_index, token in enumerate(tokens):
            token_len = len(token)
            char_offsets[token_index] = (offset, offset + token_len)
            offset += len(token) + 1
        return char_offsets

    def read_from_file(self, path):
        annotations = {}

        with open(path) as f_in:
            sentence_id: int = 0
            tokens: List[str] = []
            predicate_indices: List[int] = []
            predicate_sense: str = None
            current_role: str = None
            predicate_roles: List[str] = []
            role_start_index: int = 0

            for line in f_in:
                if line[0] == '#':
                    continue

                line = line.strip()

                if not line:
                    predicate_indices = tuple(p for p in predicate_indices)
                    annotations[sentence_id] = {}
                    annotations[sentence_id][predicate_indices] = {
                        'tokens': tokens,
                        'sense': predicate_sense,
                        'roles': predicate_roles,
                    }

                    sentence_id += 1
                    tokens = []
                    predicate_indices = []
                    predicate_sense = None
                    current_role = None
                    predicate_roles = []
                    role_start_index = 0
                    continue

                line_parts = line.split('\t')
                token_id, token, _, predicted_lemma, _, predicted_pos, _, _, _, _, _, _, predicate, sense, role = line_parts
                token_id = int(token_id) - 1
                tokens.append(self.clean_word(token[2:-1]))

                role_type, role = role[0], role[2:]

                if current_role is not None and (role_type == 'B' or role_type == 'S' or role_type == 'O'):
                    predicate_roles.append({
                        'role': current_role,
                        'start': role_start_index,
                        'end': token_id,
                    })
                    current_role = None
                
                if role_type == 'S':
                    predicate_roles.append({
                        'role': role,
                        'start': token_id,
                        'end': token_id + 1,
                    })
                    current_role = None
                
                elif role_type == 'B':
                    current_role = role
                    role_start_index = token_id

                is_predicate = sense != '_'

                if is_predicate:
                    predicate_sense = sense
                    predicate_indices.append(token_id)
    
        return annotations

    def run_scorer(self, frames_only: bool = False):
        if not frames_only:
            arguments = ['perl', self.scorer_path, '-l', '-e', '-n', self.frames_path, self.relations_path, self.gold_output_path, self.pred_output_path]
        else:
            arguments = ['perl', self.scorer_path, '-l', '-e', '-n', '-t', self.frames_path, self.relations_path, self.gold_output_path, self.pred_output_path]
        scorer_output = subprocess.run(arguments, capture_output=True, text=True).stdout

        for line in scorer_output.splitlines():
            if 'Fscore' not in line:
                continue
            
            print(line)
            line_parts = line.split()
            precision, recall, f1 = line_parts[-3], line_parts[-5], line_parts[-1]
            precision = float(precision.split('=')[-1])
            recall = float(recall.split('=')[-1])
            f1 = float(f1.split('=')[-1])

            precision_num, precision_den = line_parts[-2][1:-1].split('/')
            recall_num, recall_den = line_parts[-4][1:-1].split('/')
            break

        return {
            'precision': 100.0 * precision,
            'recall': 100.0 * recall,
            'f1': 100.0 * f1,
            'precision_num': 100.0 * float(precision_num),
            'precision_den': 100.0 * float(precision_den),
            'recall_num': 100.0 * float(recall_num),
            'recall_den': 100.0 * float(recall_den),
        }

    def build_sense_definitions_index(self) -> Dict[str, torch.Tensor]:
        print('  Building sense definitions index...')
        index: Dict[str, torch.Tensor] = {}

        for predicate_lemma, predicate_frames in tqdm(self.frames.items()):

            for predicate_sense, predicate_sense_info in predicate_frames.items():
                sense_definition = predicate_sense_info['definition']
                sense_definition = sense_definition.strip()
                if sense_definition in index:
                    continue

                sense_embedding = self.compute_embedding(sense_definition)
                index[sense_definition] = sense_embedding

        num_definitions = len(index)
        print(f'  Done! [# definitions: {num_definitions}]')
        return index

    def build_role_definitions_index(self) -> Dict[str, torch.Tensor]:
        print('  Building role definitions index...')
        index: Dict[str, torch.Tensor] = {}

        for predicate_lemma, predicate_frames in tqdm(self.frames.items()):

            for predicate_sense, predicate_sense_info in predicate_frames.items():
                roleset = predicate_sense_info['roleset']

                for role, role_definition in roleset['core'].items():
                    role_definition = role_definition.strip()
                    if role_definition in index:
                        continue

                    role_embedding = self.compute_embedding(role_definition)
                    index[role_definition] = role_embedding
                
                for role, role_definition in roleset['non-core'].items():
                    role_definition = role_definition.strip()
                    if role_definition in index:
                        continue

                    role_embedding = self.compute_embedding(role_definition)
                    index[role_definition] = role_embedding

        num_definitions = len(index)
        print(f'  Done! [# definitions: {num_definitions}]')
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
    
    def clean_word(self, word: str):
        if word == "n\'t":
            return 'not'
        if word == 'wo':
            return 'will'
        if word == "'ll":
            return 'will'
        if word == "'m":
            return 'am'
        if word == '``':
            return '"'
        if word == "''":
            return '"'
        if word == '/.':
            return '.'
        if word == '/-':
            return '...'
        if word == '-LRB-':
            return '('
        if word == '-RRB-':
            return ')'
        if word == '-LCB-':
            return '('
        if word == '-RCB-':
            return ')'
        if word == '[':
            return '('
        if word == '}':
            return ']'

        if '\\/' in word:
            word = word.replace('\\/', '/')
        
        return word
