import argparse
import json
from typing import List, Iterator, Dict, Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-datasets", nargs="+", required=True, help="The datasets to be merged"
    )
    parser.add_argument(
        "--compositional-tokens", nargs="+", help="The datasets to be merged"
    )
    parser.add_argument(
        "--datasets-identifier",
        nargs="+",
        help="The dataset identifier that will be a field in each instance of the dataset",
    )
    parser.add_argument(
        "--apply-to-source",
        action="store_true",
        help="Whether or not to use the compositional tokens in the 'source_sequence'",
    )
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def get_instances_iterator(dataset_path: str) -> Iterator[Dict[str, Any]]:
    with open(dataset_path) as f:
        for line in f:
            yield json.loads(line.strip())


def merge_datasets(
    dataset_paths: List[str],
    datasets_identifier: List[str],
    output_path: str,
    compositional_tokens: List[List[str]] = None,
    apply_to_source: bool = False,
) -> None:
    datasets_iterators = [get_instances_iterator(dp) for dp in dataset_paths]
    with open(output_path, "w") as f:
        for i, dataset_iterator in enumerate(datasets_iterators):
            curr_dataset_identifier = datasets_identifier[i]
            curr_compositional_tokens = compositional_tokens[i] if compositional_tokens is not None else None
            for instance in dataset_iterator:
                instance["dataset_id"] = curr_dataset_identifier
                if curr_compositional_tokens is not None:
                    if apply_to_source:
                        compositional_tokens_dump = "".join(map(lambda x: f"<{x}>", curr_compositional_tokens))
                        instance["source_sequence"] = f"{compositional_tokens_dump} {instance['source_sequence']}"
                    else:
                        instance["compositional_tokens"] = curr_compositional_tokens
                f.write(json.dumps(instance))
                f.write("\n")


def main():
    args = parse_args()
    assert args.datasets_identifier or args.compositional_tokens, \
        "You should specify either the datasets identifier or the compositional tokens"
    merge_datasets(
        args.input_datasets,
        args.datasets_identifier,
        args.output_path,
        compositional_tokens=[ct.split(",") for ct in args.compositional_tokens] if args.compositional_tokens else None,
        apply_to_source=args.apply_to_source,
    )


if __name__ == "__main__":
    main()
