import argparse
import os
import subprocess


def count_lines_in_file(path):
    return int(subprocess.check_output(f"wc -l \"{path}\"", shell=True).split()[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    return parser.parse_args()


def down_sample(input_path: str) -> None:
    number_of_instances = count_lines_in_file(input_path)
    percentages = ["10%", "25%", "50%", "75%"]
    dividends = [10, 4, 2, (4/3)]

    file_parts = input_path.split(".")

    shuf_dataset_path = ".".join(file_parts[:-1] + ["shuf"] + file_parts[-1:])
    os.system("shuf {} > {}".format(input_path, shuf_dataset_path))
    for perc, div in zip(percentages, dividends):
        perc_dataset_path = ".".join(file_parts[:-1] + [perc] + file_parts[-1:])
        new_number_of_instances = round(number_of_instances / div)
        os.system("head -{} {} > {}".format(new_number_of_instances, shuf_dataset_path, perc_dataset_path))


def main():
    args = parse_args()
    down_sample(args.input_path)


if __name__ == "__main__":
    main()
