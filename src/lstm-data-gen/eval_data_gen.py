import argparse
import os
from typing import List

import sentencepiece as spm

parser = argparse.ArgumentParser()

parser.add_argument("--eval_data_root", type=str, default="data/bccwj")
parser.add_argument("--A_file", type=str, default="A.txt")
parser.add_argument("--B_file", type=str, default="B.txt")
parser.add_argument("--C_file", type=str, default="C.txt")
parser.add_argument("--D_file", type=str, default="D.txt")

parser.add_argument(
    "--spm_model", type=str, default="rnng-pytorch/data/npcmj-spm.model"
)


def make_data_for_lstm_from_txt(
    txt_data_path: str,
    sp: spm.SentencePieceProcessor,
) -> List[str]:
    with open(txt_data_path, "r") as f:
        sents = [" ".join(sp.encode(line.rstrip(), out_type=str)) for line in f]
    return sents


def main(args):
    eval_data_root = args.eval_data_root
    A_file = args.A_file
    B_file = args.B_file
    C_file = args.C_file
    D_file = args.D_file

    sp = spm.SentencePieceProcessor(args.spm_model)

    A_sents = make_data_for_lstm_from_txt(os.path.join(eval_data_root, A_file), sp)
    B_sents = make_data_for_lstm_from_txt(os.path.join(eval_data_root, B_file), sp)
    C_sents = make_data_for_lstm_from_txt(os.path.join(eval_data_root, C_file), sp)
    D_sents = make_data_for_lstm_from_txt(os.path.join(eval_data_root, D_file), sp)

    with open(
        os.path.join(
            eval_data_root, os.path.splitext(os.path.basename(A_file))[0] + ".lstm.txt"
        ),
        "w",
    ) as f:
        for sent in A_sents:
            f.write(sent)
            f.write("\n")

    with open(
        os.path.join(
            eval_data_root, os.path.splitext(os.path.basename(B_file))[0] + ".lstm.txt"
        ),
        "w",
    ) as f:
        for sent in B_sents:
            f.write(sent)
            f.write("\n")

    with open(
        os.path.join(
            eval_data_root, os.path.splitext(os.path.basename(C_file))[0] + ".lstm.txt"
        ),
        "w",
    ) as f:
        for sent in C_sents:
            f.write(sent)
            f.write("\n")

    with open(
        os.path.join(
            eval_data_root, os.path.splitext(os.path.basename(D_file))[0] + ".lstm.txt"
        ),
        "w",
    ) as f:
        for sent in D_sents:
            f.write(sent)
            f.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
