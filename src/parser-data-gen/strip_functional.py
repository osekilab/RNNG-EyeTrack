import re
from typing import List, Optional

from nltk import Tree


def _remove_wrapper(tree: Tree) -> Tree:
    tree = tree[0]
    return tree


def _remove_thematic_role(tag: str) -> str:
    tag = re.sub(r"([^-]+?);.*", r"\1", tag)
    return tag


def _remove_func_tags(tag: str) -> str:
    if tag == "multi-sentence":
        return tag

    tag = re.sub(r"([^-]+?)-.*", r"\1", tag)
    return tag


def _remove_tags(tag: str) -> str:
    tag = _remove_thematic_role(tag)
    tag = _remove_func_tags(tag)
    return tag


def _remove_asterisk(tree: Tree) -> Optional[Tree]:
    if isinstance(tree, Tree) and not isinstance(tree[0], Tree):
        token = tree[0]
        if token[0] == "*":
            return None
        elif ("（" in token) or ("）" in token):
            return Tree(
                tree.label(), [token.replace("（", "-LRB-").replace("）", "-RRB-")]
            )
        else:
            return tree

    new_children = []
    for child in tree:
        new_child = _remove_asterisk(child)
        if new_child is not None:
            new_children.append(new_child)

    if len(new_children) == 0:
        return None
    else:
        return Tree(_remove_tags(tree.label()), new_children)


def _preprocess_tree(tree: Tree) -> Tree:
    tree = _remove_wrapper(tree)
    tree = _remove_asterisk(tree)
    return tree


def preprocess_trees(trees: List[Tree]) -> List[Tree]:
    preprocessed_trees = []
    error_trees = []
    for tree in trees:
        try:
            preprocessed_tree = _preprocess_tree(tree)
            preprocessed_trees.append(preprocessed_tree)
        except RecursionError as e:
            error_trees.append(tree)
            print(e)
    return preprocessed_trees, error_trees
