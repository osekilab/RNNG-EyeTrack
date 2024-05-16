import argparse
import glob
import os

from nltk import Tree
from strip_functional import preprocess_trees
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--file_root", type=str, default="data/npcmj")
parser.add_argument("--save_root", type=str, default="data/npcmj_preprocessed")
parser.add_argument("--error_root", type=str, default="data/npcmj_error")


def main(args):
    file_root = args.file_root
    save_root = args.save_root
    error_root = args.error_root
    os.makedirs(save_root, exist_ok=True)
    os.makedirs(error_root, exist_ok=True)

    file_paths = sorted(glob.glob(os.path.join(file_root, "*.psd")))
    for file_path in tqdm(file_paths):
        with open(file_path) as f:
            trees = [Tree.fromstring(line) for line in f]
        preprocessed_trees, error_trees = preprocess_trees(trees)

        save_path = os.path.join(save_root, os.path.basename(file_path))
        with open(save_path, "w") as f:
            for tree in preprocessed_trees:
                if tree is not None:
                    tree_rep = tree.pformat(margin=1e100)
                    assert "\n" not in tree_rep
                    f.write(tree_rep)
                    f.write("\n")
        if error_trees:
            error_path = os.path.join(error_root, os.path.basename(file_path))
            with open(error_path, "w") as f:
                for tree in error_trees:
                    if tree is not None:
                        tree_rep = tree.pformat(margin=1e100)
                        assert "\n" not in tree_rep
                        f.write(tree_rep)
                        f.write("\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
