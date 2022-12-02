import json

from argparse import ArgumentParser
from typing import Dict, List, Set


def read_framenet_file(path: str, frames: Dict, only_verbs: bool):
    data = []
    max_source_length = 0
    max_target_length = 0
    max_total_length = 0

    with open(path) as f:
        tokens: List[str] = []
        predicate_indices: List[int] = []
        predicate_lemma: str = None
        predicate_sense: str = None
        predicate_type: str = None
        predicate_roles: List[str] = []
        predicate_type_set: Set[str] = set()
        sentence_id: int = 0

        for line_no, line in enumerate(f):
            if line[0] == '#':
                continue
            
            line = line.strip()

            # End of sentence or end of file.
            if not line:
                if predicate_type != 'v' and only_verbs:
                    continue
                
                predicate_type_set.add(predicate_type)
                source_sequence = [t for t in tokens]
                offset = 0
                
                for i in range(len(predicate_indices)):
                    predicate_index = predicate_indices[i]
                    if i == 0 or predicate_indices[i - 1] != predicate_index - 1:
                        source_sequence = source_sequence[:predicate_index + offset] + \
                            ['<p>'] + source_sequence[predicate_index + offset:]
                        offset += 1
                    if i == len(predicate_indices) - 1 or  predicate_indices[i + 1] != predicate_index + 1:
                        source_sequence = source_sequence[:predicate_index + offset + 1] + \
                            ['</p>'] + source_sequence[predicate_index + offset + 1:]
                        offset += 1

                frame_info = frames[f'{predicate_lemma}.{predicate_type}'][predicate_sense]
                sense_definition = frame_info['definition']
                sense_definition = f'{predicate_lemma}: {sense_definition}'
                target_sequence = [t for t in tokens]
                previous_role = None
                role_definition = None
                offset = 0

                for argument_index, argument_role in enumerate(predicate_roles):
                    role_type, current_role = argument_role[0], argument_role[2:]
                    
                    if previous_role is not None and (role_type == 'B' or role_type == 'S' or role_type == 'O' or (role_type == 'I' and argument_index == len(predicate_roles) - 1)):
                        if previous_role in frame_info['roleset']['core']:
                            role_definition = frame_info['roleset']['core'][previous_role]
                        elif previous_role in frame_info['roleset']['non-core']:
                            role_definition = frame_info['roleset']['non-core'][previous_role]
                        else:
                            print(f'  Role {previous_role} not found in roleset for {predicate_lemma}.{predicate_type} in frame {predicate_sense}.')
                        
                        if role_definition:
                            target_sequence = target_sequence[:argument_start_index + offset] + \
                                ['['] + target_sequence[argument_start_index + offset : argument_index + offset] + [']{'] + \
                                [f'{role_definition}' + ' }'] + \
                                target_sequence[argument_index + offset:]
                            offset += 3
                        
                        previous_role = None
                        role_definition = None
                    
                    if role_type == 'S':
                        if current_role in frame_info['roleset']['core']:
                            role_definition = frame_info['roleset']['core'][current_role]
                        elif current_role in frame_info['roleset']['non-core']:
                            role_definition = frame_info['roleset']['non-core'][current_role]
                        else:
                            print(f'  Role {previous_role} not found in roleset for {predicate_lemma}.{predicate_type} in frame {predicate_sense}.')
                        
                        if role_definition:
                            target_sequence = target_sequence[:argument_index + offset] + \
                                ['[', target_sequence[argument_index + offset], ']{'] + \
                                [f'{role_definition}' + ' }'] + \
                                target_sequence[argument_index + offset + 1:]
                            offset += 3
                        
                        previous_role = None
                        role_definition = None
                    
                    elif role_type == 'B':
                        argument_start_index = argument_index
                        previous_role = current_role

                target_sequence = [f'{sense_definition}.'] + target_sequence

                data.append({
                    'id': sentence_id,
                    'source_sequence': ' '.join(source_sequence),
                    'target_sequence': ' '.join(target_sequence),
                    'lexical_unit': lexical_unit,
                    'predicate_indices': predicate_indices,
                    'predicate_type': predicate_type,
                    'predicate_sense': predicate_sense,
                })

                max_source_length = max(max_source_length, len(source_sequence))
                max_target_length = max(max_target_length, len(' '.join(target_sequence).split()))
                max_total_length = max(max_total_length, len(source_sequence) + len(' '.join(target_sequence).split()))

                sentence_id += 1
                tokens: List[str] = []
                predicate_indices: List[int] = []
                lexical_unit: str = None
                predicate_lemma: str = None
                predicate_sense: str = None
                predicate_type: str = None
                predicate_roles: List[str] = []
                continue

            line_parts = line.split('\t')
            token_id, token, _, predicted_lemma, _, predicted_pos, _, _, _, _, _, _, predicate, _predicate_sense, role = line_parts
            tokens.append(clean_word(token[2:-1]))
            predicate_roles.append(role)

            token_id = int(token_id) - 1
            is_predicate = _predicate_sense != '_'

            if is_predicate:
                lexical_unit = predicate
                predicate_lemma, predicate_type = predicate.split('.')
                predicate_sense = _predicate_sense
                predicate_indices.append(token_id)

    print(' Predicate types:', sorted(list(predicate_type_set)))
    print(' Max source sequence length:', max_source_length)
    print(' Max target sequence length:', max_target_length)
    print(' Max total sequence length:', max_total_length)

    return data


def clean_word(word: str):
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


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default='data_share/framenet17/original/fn1.7.dev.syntaxnet.conll')
    parser.add_argument('--frames', type=str, default='data_share/framenet17/frames/framenet_frames.json')
    parser.add_argument('--output', type=str, default='data_share/framenet17/jsonl/FrameNet17_dev.framenet17')
    parser.add_argument('--only_verbs', action='store_true')

    args = parser.parse_args()
    input_path: str = args.input
    frames_path: str = args.frames
    output_path: str = args.output
    only_verbs: bool = args.only_verbs

    print(f' Reading frames from {frames_path}...')
    with open(frames_path) as f:
        frames = json.load(f)

    print(f' Reading {input_path}...')
    data = read_framenet_file(input_path, frames, only_verbs)

    print(f' Writing data to {output_path}...')
    with open(output_path, 'w') as f:
        for instance in data:
            json_line = json.dumps(instance, sort_keys=True)
            f.write(f'{json_line}\n')

    print(' Done!\n')
