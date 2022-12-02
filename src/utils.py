import subprocess


def count_lines_in_file(path):
    return int(subprocess.check_output(f"wc -l \"{path}\"", shell=True).split()[0])
