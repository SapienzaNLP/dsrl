import os.path
import sys
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, List, Tuple, Set
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class PerformanceAnalyzer:
    def arg_analysis(self, conll_pred_path: str, conll_gold_path: str):
        with open(conll_gold_path) as g:
            gold = [l for l in g.readlines()]
        with open(conll_pred_path) as p:
            pred = [l for l in p.readlines()]
        arg_tp = {}
        arg_fp = {}
        arg_fn = {}
        for g, p in zip(gold, pred):
            if g.strip() == "":
                continue
            g = g.strip().split("\t")
            p = p.strip().split("\t")
            g_args = self.get_args_2009(g)
            p_args = self.get_args_2009(p)
            for tp_arg in [arg for arg in p_args if arg in g_args]:
                arg_tp[tp_arg[1]] = arg_tp.get(tp_arg[1], 0) + 1
            for fp_arg in [arg for arg in p_args if arg not in g_args]:
                arg_fp[fp_arg[1]] = arg_fp.get(fp_arg[1], 0) + 1
            for fn_arg in [arg for arg in g_args if arg not in p_args]:
                arg_fn[fn_arg[1]] = arg_fn.get(fn_arg[1], 0) + 1
        self.print_arg_result_table(arg_fn, arg_fp, arg_tp)

    def print_arg_result_table(
            self, arg_fn: Dict[str, int], arg_fp: Dict[str, int], arg_tp: Dict[str, int]
    ):
        print("-------------------------------")
        print("ARGUMENT LABELS RESULTS")
        print("ARG\tPRECISION\tRECALL\tF1\tSUPPORT")
        for arg, tp in sorted(arg_tp.items(), key=lambda x: x[0]):
            fp = arg_fp.get(arg, 0)
            fn = arg_fn.get(arg, 0)
            precision = float(tp) / (tp + fp)
            recall = float(tp / (tp + fn))
            f1 = 2 * precision * recall / (precision + recall)
            support = tp + fn
            print(
                f"{arg}\t{round(precision, 3)}\t{round(recall, 3)}\t{round(f1, 3)}\t{support}"
            )
        for arg in set(
                [arg for arg in arg_fp.items() if arg[0] not in arg_tp]
                + [arg for arg in arg_fn.items() if arg[0] not in arg_tp]
        ):
            support = arg_fn.get(arg[0], 0)
            print(f"{arg[0]}\t{0.0}\t{0.0}\t{0.0}\t{support}")
        print("")

    def get_args_2009(self, line: List[str]):
        return [(i, arg, line[0]) for i, arg in enumerate(line[14:]) if arg != "_"]

    def get_args_2012(self, line: List[str]):
        args = [(i, arg.replace("*", ''), line[2]) for i, arg in enumerate(line[11:-1]) if
                arg != "*"]
        return [arg for arg in args if arg[1] != "(V)"]

    def predicate_analysis(
            self,
            conll_pred_path: str,
            conll_gold_path: str,
            dataset: str,
            conll_train_path: str,
            conll2009_match: bool,

    ):
        train_dir = Path(conll_train_path).parent
        senses_mfl_lfs_path = os.path.join(train_dir, "predicate_mfs.tsv")
        mfs = {}
        t = open(conll_train_path)
        if dataset != "conll2009":
            next(t)
        chunks = [l.split("\t") for l in t.readlines() if l.strip() != ""] if dataset == "conll2009" \
            else [l.split() for l in t.readlines() if l.strip() != "" and not l.startswith("#")]
        seen_senses = [chunk[13] for chunk in chunks] if dataset == "conll2009" \
            else [chunk[6] + "." + chunk[7] for chunk in chunks if chunk[7] != '-']
        seen_senses = [sense for sense in seen_senses if sense.strip() != "_"]
        sense_pairs = [sense.split(".") for sense in seen_senses]
        if not os.path.exists(senses_mfl_lfs_path):
            self.compute_mfs_lfs(mfs, sense_pairs, senses_mfl_lfs_path)
        else:
            self.parse_mfs_lfs(mfs, senses_mfl_lfs_path)
        seen_senses = set(seen_senses)
        with open(conll_gold_path) as g:
            gold = [l for l in g.readlines() if not l.strip().startswith("#")]
        with open(conll_pred_path) as p:
            pred = [l for l in p.readlines() if not l.strip().startswith("#")]
        all_ids, mfs_ids, lfs_ids, unseen_ids, predicted_args, gold_args = [], [], [], [], [], []
        self.pred_id = 0
        counters = {}
        for g, p in zip(gold, pred):
            if g.strip() == "":
                if dataset == "conll2012":
                    predicted_args = self.aggregate_span_args(predicted_args)
                    gold_args = self.aggregate_span_args(gold_args)


                counters = self.update_counters(
                    counters,
                    "MFS",
                    gold_args,
                    mfs_ids,
                    predicted_args,
                )
                counters = self.update_counters(
                    counters,
                    "ALL",
                    gold_args,
                    all_ids,
                    predicted_args,
                )
                counters = self.update_counters(
                    counters,
                    "LFS",
                    gold_args,
                    lfs_ids,
                    predicted_args,
                )
                counters = self.update_counters(
                    counters,
                    "UNSEEN",
                    gold_args,
                    unseen_ids,
                    predicted_args,
                )
                all_ids, mfs_ids, lfs_ids, unseen_ids, predicted_args, gold_args = (
                    [],
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                self.pred_id = 0
                continue
            counters = self.parse_conll_info(
                dataset,
                g, p, gold_args, predicted_args, mfs, all_ids, mfs_ids, lfs_ids, unseen_ids, seen_senses, counters,
                conll2009_match
            )
        self.print_result_table(counters)

    def aggregate_span_args(self, args):
        predid2_args = defaultdict(list)
        for arg in args:
            predid2_args[arg[0]].append((arg[1], arg[2]))
        final_args = []
        for pred_id, args in predid2_args.items():
            span_arg = None
            for arg_label, tok_id in args:
                if arg_label.startswith('(') and arg_label.endswith(')'):
                    final_args.append((pred_id, arg_label, tok_id))
                elif arg_label.startswith('('):
                    span_arg = (arg_label, tok_id)
                else:
                    span_arg = (span_arg[0] + arg_label, span_arg[1] + "-" + tok_id)
                    final_args.append((pred_id, span_arg[0], span_arg[1]))
                    span_arg = None
        return final_args

    def print_result_table(
            self, counters: Dict[str, Dict[str, int]]
    ):
        print("-------------------------------")
        print("PREDICATE SENSE DISAMBIGUATION")
        print("\tACCURACY\tSUPPORT")
        for partition_key in counters:
            print(
                f"{partition_key}\t{round(float(counters[partition_key]['CORRECT']) / counters[partition_key]['TOTAL'], 3)}\t{counters[partition_key]['TOTAL']}"
            )
        print("-------------------------------")
        print("ARGUMENT LABELING")
        print("\tPRECISION\tRECALL\tF1\tSUPPORT")
        for partition_key in counters:
            arg_precision = float(counters[partition_key]['TP']) / (
                    counters[partition_key]['TP'] + counters[partition_key]['FP'])
            arg_recall = float(counters[partition_key]['TP']) / (
                    counters[partition_key]['TP'] + counters[partition_key]['FN'])
            arg_f1 = (
                    2
                    * arg_precision
                    * arg_recall
                    / (arg_precision + arg_recall)
            )
            arg_support = counters[partition_key]['TP'] + counters[partition_key]['FN']
            print(
                f"{partition_key}\t{round(arg_precision, 3)}\t{round(arg_recall, 3)}\t{round(arg_f1, 3)}\t{arg_support}"
            )

    def parse_conll_info(
            self,
            dataset: str,
            g: str,
            p: str,
            gold_args: List[Tuple[int, str, str]],
            predicted_args: List[Tuple[int, str, str]],
            mfs: Dict[str, str],
            all_ids: List[int],
            mfs_ids: List[int],
            lfs_ids: List[int],
            unseen_ids: List[int],
            seen_senses: Set[str],
            counters: Dict[str, Dict[str, int]],
            conll2009_match: bool = False
    ):

        g = g.strip().split("\t") if dataset == "conll2009" else g.strip().split()
        p = p.strip().split("\t") if dataset == "conll2009" else p.strip().split()
        p_predicate = p[13] if dataset == "conll2009" else f"{p[6]}.{p[7]}"
        g_predicate = g[13] if dataset == "conll2009" else f"{p[6]}.{p[7]}"
        p_args = self.get_args_2009(p) if dataset == "conll2009" else self.get_args_2012(p)
        g_args = self.get_args_2009(g) if dataset == "conll2009" else self.get_args_2012(g)
        for p_arg in p_args:
            predicted_args.append(p_arg)
        for g_arg in g_args:
            gold_args.append(g_arg)
        if "." in g_predicate and "-" not in g_predicate:
            pred_sense_number = p_predicate.split(".")[-1]
            g_sense_number = g_predicate.split(".")[-1]
            matched = g_predicate == p_predicate if not conll2009_match else pred_sense_number == g_sense_number
            g_lemma, g_id = g_predicate.split(".")
            all_ids.append(self.pred_id)
            if matched:
                counters = self.increase_counter(counters, "ALL", "CORRECT")
            counters = self.increase_counter(counters, "ALL", "TOTAL")
            if g_predicate not in seen_senses:
                unseen_ids.append(self.pred_id)
                if matched:
                    counters = self.increase_counter(counters, "UNSEEN", "CORRECT")
                counters = self.increase_counter(counters, "UNSEEN", "TOTAL")
            elif mfs[g_lemma] == g_id:
                mfs_ids.append(self.pred_id)
                if matched:
                    counters = self.increase_counter(counters, "MFS", "CORRECT")
                counters = self.increase_counter(counters, "MFS", "TOTAL")
            else:
                lfs_ids.append(self.pred_id)
                if matched:
                    counters = self.increase_counter(counters, "LFS", "CORRECT")
                counters = self.increase_counter(counters, "LFS", "TOTAL")
            self.pred_id += 1
        return counters

    def increase_counter(self, counters: Dict[str, Dict[str, int]], partition_key: str, field: str, amount: int = 1):
        partition_counters = counters.get(partition_key, {})
        partition_counters[field] = partition_counters.get(field, 0) + amount
        counters[partition_key] = partition_counters
        return counters

    def update_counters(
            self,
            counters,
            partition_key,
            g_arg_pred_ids,
            sense_ids,
            p_arg_pred_ids,
    ):
        p_args = [arg for arg in p_arg_pred_ids if arg[0] in sense_ids]
        g_args = [arg for arg in g_arg_pred_ids if arg[0] in sense_ids]
        counters = self.increase_counter(counters, partition_key, "TP", len([arg for arg in p_args if arg in g_args]))
        counters = self.increase_counter(counters, partition_key, "FP",
                                         len([arg for arg in p_args if arg not in g_args]))
        counters = self.increase_counter(counters, partition_key, "FN",
                                         len([arg for arg in g_args if arg not in p_args]))
        return counters

    def parse_mfs_lfs(self, mfs, senses_mfl_lfs_path):
        f = open(senses_mfl_lfs_path)
        next(f)
        for line in f:
            line = line.strip().split("\t")
            mfs[line[0]] = line[1]

    def compute_mfs_lfs(self, mfs, sense_pairs, senses_mfl_lfs_path):
        with open(senses_mfl_lfs_path, "w") as f:
            f.write("PREDICATE\tMFS\n")
            for p in tqdm(
                    set([sense[0] for sense in sense_pairs]), "computing MFS"
            ):
                senses_id = [sense[1] for sense in sense_pairs if sense[0] == p]
                mf_sense = max(Counter(senses_id).items(), key=lambda x: x[1])[0]
                mfs[p] = mf_sense
            for predicate in mfs:
                f.write(f'{predicate}\t{mfs[predicate]}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", type=str, required=True)
    parser.add_argument("-g", type=str, required=True)
    # required to compute MFS
    parser.add_argument(
        "--train_dataset_path", type=str, required=True)
    parser.add_argument("--conll2009_predicate_match", default=False, action="store_true")
    args = parser.parse_args()
    system_annotations: str = args.s
    gold_annotations: str = args.g
    dataset: str = "conll2009"
    train_dataset_path: str = args.train_dataset_path
    analyzer = PerformanceAnalyzer()
    analyzer.predicate_analysis(system_annotations, gold_annotations, dataset, train_dataset_path,
                                args.conll2009_predicate_match)
    analyzer.arg_analysis(system_annotations, gold_annotations)
