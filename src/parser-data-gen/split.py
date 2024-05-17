import argparse
import glob
import os

from nltk import Tree
from sklearn.model_selection import train_test_split
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--preprocessed_root", type=str, default="data/npcmj_preprocessed")
parser.add_argument("--save_root", type=str, default="data/npcmj_split")


def main(args):
    preprocessed_root = args.preprocessed_root
    save_root = args.save_root
    os.makedirs(save_root, exist_ok=True)

    file_list = sorted(glob.glob(os.path.join(preprocessed_root, "*.psd")))
    random_state = 42

    aozora = []
    bible = []
    blog = []
    book = []
    dictionary = []
    diet = []
    essay = []
    fiction = []
    law = []
    misc = []
    news = []
    nonfiction = []
    spoken = []
    ted = []
    textbook = []
    wikipedia = []
    patent = []
    whitepaper = []
    stories = []
    for filename in tqdm(file_list):
        if "aozora" in filename:
            with open(filename) as f:
                for line in f:
                    aozora.append(line.rstrip())
        elif "bible" in filename:
            with open(filename) as f:
                for line in f:
                    bible.append(line.rstrip())
        elif "blog" in filename:
            with open(filename) as f:
                for line in f:
                    blog.append(line.rstrip())
        elif "textbook" in filename:
            with open(filename) as f:
                for line in f:
                    textbook.append(line.rstrip())
        elif "book" in filename:
            with open(filename) as f:
                for line in f:
                    book.append(line.rstrip())
        elif "dict" in filename:
            with open(filename) as f:
                for line in f:
                    dictionary.append(line.rstrip())
        elif "diet" in filename:
            with open(filename) as f:
                for line in f:
                    diet.append(line.rstrip())
        elif "essay" in filename:
            with open(filename) as f:
                for line in f:
                    essay.append(line.rstrip())
        elif "law" in filename:
            with open(filename) as f:
                for line in f:
                    law.append(line.rstrip())
        elif "misc" in filename:
            with open(filename) as f:
                for line in f:
                    misc.append(line.rstrip())
        elif "news" in filename:
            with open(filename) as f:
                for line in f:
                    news.append(line.rstrip())
        elif "nonfiction" in filename:
            with open(filename) as f:
                for line in f:
                    nonfiction.append(line.rstrip())
        elif "fiction" in filename:
            with open(filename) as f:
                for line in f:
                    fiction.append(line.rstrip())
        elif "spoken" in filename:
            with open(filename) as f:
                for line in f:
                    spoken.append(line.rstrip())
        elif "ted" in filename:
            with open(filename) as f:
                for line in f:
                    ted.append(line.rstrip())
        elif "wikipedia" in filename:
            with open(filename) as f:
                for line in f:
                    wikipedia.append(line.rstrip())
        elif "patent" in filename:
            with open(filename) as f:
                for line in f:
                    patent.append(line.rstrip())
        elif "whitepaper" in filename:
            with open(filename) as f:
                for line in f:
                    whitepaper.append(line.rstrip())
        elif "stories" in filename:
            with open(filename) as f:
                for line in f:
                    stories.append(line.rstrip())
        else:
            print("Error: {}".format(filename))

    all_tree = [
        aozora,
        bible,
        blog,
        book,
        dictionary,
        diet,
        essay,
        fiction,
        law,
        misc,
        news,
        nonfiction,
        spoken,
        ted,
        textbook,
        wikipedia,
        patent,
        whitepaper,
        stories,
    ]

    trains = []
    vals = []
    tests = []
    for file_tree in tqdm(all_tree):
        train, val = train_test_split(
            file_tree, test_size=0.1, random_state=random_state
        )
        val, test = train_test_split(val, test_size=0.5, random_state=random_state)
        for tree in train:
            trains.append(tree)
        for tree in val:
            vals.append(tree)
        for tree in test:
            tests.append(tree)

    print(
        len(trains),
        len(vals),
        len(tests),
        sum([len(trains), len(vals), len(tests)]),
    )

    with open(os.path.join(save_root, "train.mrg"), "w") as f:
        for tree in trains:
            assert "\n" not in tree
            f.write(tree)
            f.write("\n")

    with open(os.path.join(save_root, "dev.mrg"), "w") as f:
        for tree in vals:
            assert "\n" not in tree
            f.write(tree)
            f.write("\n")

    with open(os.path.join(save_root, "test.mrg"), "w") as f:
        for tree in tests:
            assert "\n" not in tree
            f.write(tree)
            f.write("\n")

    def extract_leaves_from_trees(trees):
        leaves = [" ".join(Tree.fromstring(tree).leaves()) for tree in trees]
        return leaves

    train_leaves = extract_leaves_from_trees(trains)
    val_leaves = extract_leaves_from_trees(vals)
    test_leaves = extract_leaves_from_trees(tests)

    train_leaves_str = "\n".join(train_leaves)
    val_leaves_str = "\n".join(val_leaves)
    test_leaves_str = "\n".join(test_leaves)

    with open(os.path.join(save_root, "train.txt"), "w") as f:
        f.write(train_leaves_str)
    with open(os.path.join(save_root, "val.txt"), "w") as f:
        f.write(val_leaves_str)
    with open(os.path.join(save_root, "test.txt"), "w") as f:
        f.write(test_leaves_str)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
