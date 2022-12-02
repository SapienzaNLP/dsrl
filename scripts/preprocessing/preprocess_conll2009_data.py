import json

from argparse import ArgumentParser
from collections import Counter
from typing import Dict, List


def read_conll2009_file(path: str, frames: dict, argm_definitions: dict, only_verbs: bool):
    verb_pos_set = {'VBD', 'VBZ', 'VBG', 'VBP', 'VBN', 'MD', 'VB', 'PRF'}
    data = []
    max_source_length = 0
    max_target_length = 0
    max_total_length = 0

    with open(path) as f:
        tokens: List[str] = []
        predicate_indices: List[int] = []
        predicate_senses: List[str] = []
        predicate_types: List[str] = []
        line_roles: List[List[str]] = []
        unidentifiable_senses: List[str] = []
        unidentifiable_roles: List[str] = []
        sentence_id: int = 0

        for line_no, line in enumerate(f):
            line = line.strip()

            # End of sentence or end of file.
            if not line:
                predicate_roles = list(map(list, zip(*line_roles)))
                predicate_info = zip(predicate_indices, predicate_senses, predicate_types, predicate_roles)
                for predicate_index, predicate_sense, predicate_type, predicate_roles in predicate_info:
                    if predicate_type == 'n' and only_verbs:
                        continue

                    source_sequence = tokens[:predicate_index] + ['<p>', tokens[predicate_index], '</p>'] + tokens[predicate_index + 1:]

                    predicate_lemma, predicate_sense_number = predicate_sense.split('.')
                    if predicate_lemma not in frames:
                        # print(f' Predicate lemma {predicate_lemma} not in available frames...')
                        unidentifiable_senses.append(predicate_sense)
                        sense_definition = f'{predicate_lemma}: {predicate_lemma}'
                        predicate_roleset = {'A0': 'agent', 'A1': 'theme'}

                    else:
                        predicate_frames = frames[predicate_lemma]
                        predicate_sense = f'{predicate_lemma}.{predicate_type}.{predicate_sense_number}'
                        if predicate_type == 'n' and predicate_sense not in predicate_frames and f'{predicate_lemma}.v.{predicate_sense_number}' in predicate_frames:
                            predicate_type = 'v'
                            predicate_sense = f'{predicate_lemma}.v.{predicate_sense_number}'

                        if predicate_sense not in predicate_frames:
                            # print(f' Predicate sense {predicate_sense} not in available senses...')
                            unidentifiable_senses.append(predicate_sense)
                            sense_definition = f'{predicate_lemma}: {predicate_lemma}'
                            predicate_roleset = {'A0': 'agent', 'A1': 'theme'}
                        
                        else:
                            sense_definition = predicate_frames[predicate_sense]['definition']
                            predicate_roleset = predicate_frames[predicate_sense]['roleset']

                    offset = 0
                    target_sequence = [t for t in tokens]
                    
                    for argument_index, argument_role in enumerate(predicate_roles):
                        if argument_role in {'_', 'AM-PRD', 'AM-REC', 'AM', 'AA', 'AM-PRT'}:
                            continue

                        if argument_role[:2] == 'R-':
                            role_prefix = '<reference-to> '
                            _argument_role = argument_role[2:]

                        elif argument_role[:2] == 'C-':
                            role_prefix = '<continuation-of> '
                            _argument_role = argument_role[2:]
                        else:
                            role_prefix = ''
                            _argument_role = argument_role

                        if _argument_role in predicate_roleset:
                            role_definition = predicate_roleset[_argument_role]
                        elif _argument_role in argm_definitions:
                            role_definition = argm_definitions[_argument_role]
                        else:
                            unidentifiable_roles.append(_argument_role)
                            continue

                        target_sequence = target_sequence[:argument_index + offset] + \
                            [target_sequence[argument_index + offset]] + \
                            ['{ ' + f'{role_prefix}{role_definition}' + ' }'] + \
                            target_sequence[argument_index + offset + 1:]
                        offset += 1
                    
                    target_sequence = [f'{sense_definition}.'] + target_sequence

                    data.append({
                        'id': sentence_id,
                        'source_sequence': ' '.join(source_sequence),
                        'target_sequence': ' '.join(target_sequence),
                        'predicate_indices': [predicate_index],
                        'predicate_type': predicate_type
                    })
                    
                    max_source_length = max(max_source_length, len(source_sequence))
                    max_target_length = max(max_target_length, len(' '.join(target_sequence).split()))
                    max_total_length = max(max_total_length, len(source_sequence) + len(' '.join(target_sequence).split()))


                sentence_id += 1
                tokens: List[str] = []
                predicate_indices: List[int] = []
                predicate_senses: List[str] = []
                predicate_types: List[str] = []
                line_roles: List[List[str]] = []
                continue

            line_parts = line.split()
            token_id, token, g_lemma, p_lemma, g_pos, p_pos, _, _, g_head, p_head, g_deprel, p_deprel, is_predicate, predicate_sense, *roles = line_parts
            token = clean_word(token)
            tokens.append(token)

            token_id = int(token_id) - 1
            is_predicate = is_predicate == 'Y'

            if is_predicate:
                predicate_type = 'v' if g_pos in verb_pos_set else 'n'
                predicate_indices.append(token_id)
                predicate_senses.append(predicate_sense)
                predicate_types.append(predicate_type)

            line_roles.append(roles)

    print(' # senses that could not be found:', len(unidentifiable_senses))
    print(' # roles that could not be mapped:', len(unidentifiable_roles))
    print(' Max source sequence length:', max_source_length)
    print(' Max target sequence length:', max_target_length)
    print(' Max total sequence length:', max_total_length)

    return data


def clean_word(word: str):
    if word == "n\'t":
        return 'not'
    if word == 'wo':
        return 'will'
    if word == '``':
        return '"'
    if word == "''":
        return '"'
    if word == '{':
        return '('
    if word == '}':
        return ')'

    if '\\/' in word:
        word = word.replace('\\/', '/')
    
    return word


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='data_share/conll2009/original/CoNLL2009_test-ood.txt')
    parser.add_argument('--frames', type=str, default='data_share/conll2009/frames/conll2009_frames.json')
    parser.add_argument('--argm_definitions', type=str,
                        default='data_share/conll2009/frames/conll2009_argm_definitions.json')
    parser.add_argument('--output', type=str, default='data_share/conll2009/jsonl/CoNLL2009_test-ood.jsonl')
    parser.add_argument('--only_verbs', action='store_true')

    args = parser.parse_args()
    input_path: str = args.input
    frames_path: str = args.frames
    argm_definitions_path: str = args.argm_definitions
    output_path: str = args.output
    only_verbs: bool = args.only_verbs

    with open(frames_path) as f:
        frames = json.load(f)

    with open(argm_definitions_path) as f:
        argm_definitions = json.load(f)

    print(f' Reading {input_path}...')
    data = read_conll2009_file(input_path, frames, argm_definitions, only_verbs)

    print(f' Writing data to {output_path}...')
    with open(output_path, 'w') as f:
        for instance in data:
            json_line = json.dumps(instance, sort_keys=True)
            f.write(f'{json_line}\n')

    print(' Done!\n')
