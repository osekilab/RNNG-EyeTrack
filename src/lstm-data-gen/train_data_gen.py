import argparse
import json
import os
from typing import List

parser = argparse.ArgumentParser()

parser.add_argument("--rnng_data_root", type=str, default="rnng-pytorch/data")
parser.add_argument("--train_file", type=str, default="npcmj-train.json")
parser.add_argument("--val_file", type=str, default="npcmj-val.json")
parser.add_argument("--test_file", type=str, default="npcmj-test.json")
parser.add_argument("--vocab_file", type=str, default="npcmj-spm.vocab")


def make_data_for_lstm_from_json(rnng_data_path: str) -> List[str]:
    with open(rnng_data_path, "r") as f:
        infos = [json.loads(line) for line in f.readlines()]
    sents = [" ".join(info["tokens"]) for info in infos if "tokens" in info.keys()]
    return sents


def make_vocab_for_lstm(vocab_data_path: str) -> List[str]:
    with open(vocab_data_path, "r") as f:
        vocab = ['<eos>'] + [line.split("\t")[0] for line in f.readlines()]
    return vocab


def main(args):
    rnng_data_root = args.rnng_data_root
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file

    train_sents = make_data_for_lstm_from_json(os.path.join(rnng_data_root, train_file))
    val_sents = make_data_for_lstm_from_json(os.path.join(rnng_data_root, val_file))
    test_sents = make_data_for_lstm_from_json(os.path.join(rnng_data_root, test_file))

    vocab = make_vocab_for_lstm(os.path.join(rnng_data_root, args.vocab_file))

    with open("neural-complexity/data/npcmj-train.txt", "w") as f:
        for sent in train_sents:
            f.write(sent)
            f.write("\n")
    with open("neural-complexity/data/npcmj-val.txt", "w") as f:
        for sent in val_sents:
            f.write(sent)
            f.write("\n")
    with open("neural-complexity/data/npcmj-test.txt", "w") as f:
        for sent in test_sents:
            f.write(sent)
            f.write("\n")
    with open("neural-complexity/data/npcmj-vocab.txt", "w") as f:
        for v in vocab:
            f.write(v)
            f.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
