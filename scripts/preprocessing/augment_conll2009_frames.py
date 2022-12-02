import json
from argparse import ArgumentParser


def augment_conll2009_frames(conll2009_frames, conll2012_frames):
    
    for lemma in conll2012_frames:
        if lemma not in conll2009_frames:
            conll2012_frameset = conll2012_frames[lemma]
            new_conll2009_frameset = {}

            for sense in conll2012_frameset:
                new_conll2009_frameset[sense] = {
                    'definition': conll2012_frameset[sense]['definition'],
                    'roleset': {}
                }

                for role, definition in conll2012_frameset[sense]['roleset'].items():
                    role = role.replace('ARG', 'A')
                    new_conll2009_frameset[sense]['roleset'][role] = definition

            conll2009_frames[lemma] = new_conll2009_frameset

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--conll2009_frames', type=str, default='data_share/conll2009/frames/conll2009_frames.json')
    parser.add_argument('--conll2012_frames', type=str, default='data_share/conll2012/frames/conll2012_frames.json')
    parser.add_argument('--output', type=str, default='data_share/conll2009/frames/conll2009_frames.augmented.json')

    args = parser.parse_args()
    conll2009_frames_path: str = args.conll2009_frames
    conll2012_frames_path: str = args.conll2012_frames
    output_path: str = args.output

    with open(conll2009_frames_path) as f:
        conll2009_frames = json.load(f)
    
    with open(conll2012_frames_path) as f:
        conll2012_frames = json.load(f)

    augment_conll2009_frames(conll2009_frames, conll2012_frames)

    with open(output_path, 'w') as f:
        json.dump(conll2009_frames, f, sort_keys=True, indent=2)

    print(' Done!')
