import os
from argparse import ArgumentParser
from xml.dom import minidom
from xml.etree import ElementTree as ET

ET.register_namespace('', 'http://framenet.icsi.berkeley.edu')

def merge_frames(frames_dir: str, output_path: str):
    root = ET.Element('frames')

    for filename in sorted(os.listdir(frames_dir)):
        if not filename.endswith('.xml'):
            continue

        frame_path = os.path.join(frames_dir, filename)
        frame_tree = ET.parse(frame_path)
        frame_node = frame_tree.getroot()
        root.append(frame_node)
    
    tree_string = ET.tostring(root, encoding='utf-8')
    reparsed = minidom.parseString(tree_string)
    pretty_xml = reparsed.toprettyxml(indent='  ')

    with open(output_path, 'w') as f:
        f.write(pretty_xml)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--frames_dir', type=str, default='data_share/framenet17/frames/frame_files')
    parser.add_argument('--output', type=str, default='data_share/framenet17/frames/framenet_frames.xml')

    args = parser.parse_args()
    frames_dir: str = args.frames_dir
    output_path: str = args.output

    print(f' Merging FrameNet frames in {frames_dir}...')
    merge_frames(frames_dir, output_path)

    print(' Done!')
