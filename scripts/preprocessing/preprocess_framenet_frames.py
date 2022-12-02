import os
from argparse import ArgumentParser
from typing import Dict
from xml.etree import ElementTree as ET
import re
import json


def read_frames(path: str):
    frames = {}

    for filename in sorted(os.listdir(path)):
        if not filename.endswith('.xml'):
            continue

        file_path = os.path.join(path, filename)
        file_frames = read_framefile(file_path)
        for lu in file_frames:
            if lu not in frames:
                frames[lu] = file_frames[lu]
            else:
                frames[lu].update(file_frames[lu])

    return frames


def clean_role_definitions(frame_name: str, roleset: Dict[str, str]):
    simple_frame_name = frame_name.replace('_', ' ').lower()

    for role_type in roleset:
        for role in roleset[role_type]:
            definition = roleset[role_type][role].lstrip()
            
            for other_role in roleset:
                if other_role in definition:
                    cleaned_role = other_role.replace('_', ' ').lower()
                    definition = definition.replace(other_role, cleaned_role)

            definition = re.sub(r'</?.*?>', '', definition)
            definition = re.sub(r'i\.e\. ?', '', definition)
            definition = re.sub(r'e\.g\. ?', '', definition)
            definition = re.sub(r'etc\.? ?', '', definition)
            definition = re.sub(r' \(.*?\)', '', definition)
            definition = re.sub(r'  +([a-z])', ' \\1', definition)
            definition = re.sub(r'  +(fe|FE)', ' \\1', definition)
            definition = re.sub(r'(fe|FE)(identifies)', '\\1 \\2', definition)
            definition = re.sub(r'^ *[Tt]ypically, ', '', definition)
            definition = re.sub(r'^ *[Ff]requently, ', '', definition)
            definition = re.sub(r',.*?typically.*?,', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (indicates|denotes)( (that|what))?', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (marks|(is applied to)) expressions (that|which) (indicate|denote|characterize) ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) marks expressions (that are )?', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) marks expressions indicating ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) marks constituents (which|that) express', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (designates|signifies|describes|expresses|identifies) (that )?', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (signify|describe|express|identify) ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (refers|applies) to ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) ((is defined as)|(is for)) ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (label )?is used (for|to|when) (the items are )?', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) is assigned to phrases expressing ', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) [a-z]+ (the|a|an|any) ', '', definition)
            definition = re.sub(r'^.*(([Ff]rame [Ee]lement)|([Ff][Ee])) (that [a-z]* )', '', definition)
            definition = re.sub(r'^.*([Tt]his )?(([Ff]rame [Ee]lement)|([Ff][Ee])) (is )?', '', definition)
            definition = re.sub(r'^ *[Ee]xpressions marked.*?indicate that.*as ', '', definition)
            definition = re.sub(r'^.*[Tt]here is a constituent expressing ', '', definition)
            definition = re.sub(r',.*?[Pp][Pp] [Cc]omplement.*', '', definition)
            definition = re.sub(r' fes', 'semantic roles', definition)
            definition = re.sub(r'^ *[Ff][Nn]: ', '', definition)
            definition = re.sub(r'[a-zA-Z]+ (is|(marks expressions that indicate)|(marks expressions which describe)) ', '', definition)
            definition = re.sub(r'^ *[Tt]his is ', '', definition)
            definition = re.sub(r'^ *([Tt]he|[Aa]n?) ', '', definition)
            definition = re.sub(r'( [a-zA-Z]+)\1 ', '\\1 ', definition)
            
            definition = definition.lower()
            definition = definition.replace(frame_name.lower(), f'"{simple_frame_name}"')
            definition = definition.replace('_', ' ')

            if '.' in definition:
                split_index = definition.find('.')
                definition = definition[:split_index].strip()
            if '\t' in definition:
                split_index = definition.find('\t')
                definition = definition[:split_index].strip()
            if '\n' in definition:
                split_index = definition.find('\n')
                definition = definition[:split_index].strip()
            if '  ' in definition:
                split_index = definition.find('  ')
                definition = definition[:split_index].strip()
            if ';' in definition:
                split_index = definition.find(';')
                definition = definition[:split_index].strip()
            if ':' in definition:
                split_index = definition.find(':')
                definition = definition[:split_index].strip()

            if 'selects some gradable attribute' in definition:
                definition = 'gradable attribute'

            if not definition or 'nb:' in definition:
                print(f'  Null role definition for {role} in {frame_name}...')
                definition = role.lower().strip().replace('_', ' ')
            
            definition = definition.strip()
            if definition[-1] == '.' or definition[-1] == ',':
                definition = definition[:-1]
        
            roleset[role_type][role] = definition
    
    return roleset


def read_framefile(path: str):
    frames = {}
    roleset = {
        'core': {},
        'non-core': {},
    }

    prefix = '{http://framenet.icsi.berkeley.edu}'
    tree = ET.parse(path)
    frame_node = tree.getroot()
    frame_name = frame_node.attrib['name']
    
    for element_node in frame_node.iter(f'{prefix}FE'):
        role = element_node.attrib['name']
        definition = element_node.find(f'{prefix}definition').text
        if element_node.attrib['coreType'].lower() == 'core':
            roleset['core'][role] = definition
        else:
            roleset['non-core'][role] = definition
    
    roleset = clean_role_definitions(frame_name, roleset)
    
    for lu_node in frame_node.iter(f'{prefix}lexUnit'):
        lu_name = lu_node.attrib['name']
        lu_definition = lu_node.find(f'{prefix}definition').text
        try:
            lu_definition = lu_definition.lower()
        except AttributeError:
            print(f'  Null LU definition for {lu_name} in {frame_name}...')
            lu_definition = lu_name.lower().strip().replace('_', ' ')

        lu_definition = re.sub(r'</?.*?>', '', lu_definition)
        lu_definition = re.sub(r'i\.e\. ?', '', lu_definition)
        lu_definition = re.sub(r'e\.g\. ?', '', lu_definition)
        lu_definition = re.sub(r'^ *fn: ', '', lu_definition)
        lu_definition = re.sub(r'( [a-zA-Z]+)\1 ', '\\1 ', lu_definition)
        lu_definition = re.sub(r'[:;] ?patterns like:.*', '', lu_definition)
        lu_definition = re.sub(r'; +?([a-z0-9]|\(.*?\))', ', \\1', lu_definition)
        lu_definition = re.sub(r';$', '', lu_definition)
        lu_definition = re.sub(r'  +', ' ', lu_definition)

        if ':' in lu_definition:
            split_index = lu_definition.find(':') + 1
            lu_definition = lu_definition[split_index:].strip()
        if '\n' in lu_definition:
            split_index = lu_definition.find('\n')
            lu_definition = lu_definition[:split_index].strip()
        if '.' in lu_definition:
            split_index = lu_definition.find('.')
            lu_definition = lu_definition[:split_index].strip()
        if ';' in lu_definition:
            split_index = lu_definition.find(';')
            lu_definition = lu_definition[:split_index].strip()
        if 'nb:' in lu_definition:
            split_index = lu_definition.find('nb:')
            lu_definition = lu_definition[:split_index].strip()
        
        lu_definition = lu_definition.strip()
        
        if lu_name not in frames:
            frames[lu_name] = {}
        
        frames[lu_name][frame_name] = {
            'definition': lu_definition,
            'roleset': roleset,
        }

    return frames


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--framenet_frames', type=str, default='data_share/framenet17/frames/frame_files')
    parser.add_argument('--output', type=str, default='data_share/framenet17/frames/framenet_frames.json')

    args = parser.parse_args()
    frames_path: str = args.framenet_frames
    output_path: str = args.output

    print(f' Reading FrameNet frames from {frames_path}...')
    frames = read_frames(frames_path)

    print(f' Saving to {output_path}...')
    with open(output_path, 'w') as f:
        json.dump(frames, f, sort_keys=True, indent=2)

    print(' Done!')
