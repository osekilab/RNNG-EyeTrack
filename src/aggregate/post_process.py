import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="data/aggregated/concat/fpt.csv")
parser.add_argument(
    "--save_path", type=str, default="data/aggregated/concat/fpt-del.csv"
)


def _add_prev_info(
    bccwj_eyetrack: pd.DataFrame,
    col_name: str,
    prefix: str = "prev_{}",
    article_col_name: str = "article",
) -> pd.DataFrame:
    bccwj_eyetrack[prefix.format(article_col_name)] = bccwj_eyetrack[
        article_col_name
    ].shift(1)

    is_article_first_col_name = "is_{}_first".format(article_col_name)
    bccwj_eyetrack[is_article_first_col_name] = (
        bccwj_eyetrack[article_col_name]
        != bccwj_eyetrack[prefix.format(article_col_name)]
    )

    bccwj_eyetrack[prefix.format(col_name)] = bccwj_eyetrack[col_name].shift(
        1, fill_value=0
    )
    bccwj_eyetrack.loc[
        bccwj_eyetrack[is_article_first_col_name], prefix.format(col_name)
    ] = 0

    return bccwj_eyetrack


def _remove_out_of_main_text(
    bccwj_eyetrack: pd.DataFrame,
) -> pd.DataFrame:
    return bccwj_eyetrack[bccwj_eyetrack["metadata"] == 1]


def _remove_not_fixated(
    bccwj_eyetrack: pd.DataFrame,
) -> pd.DataFrame:
    return bccwj_eyetrack[bccwj_eyetrack["logtime"] != -np.inf]


def _remove_unk(
    bccwj_eyetrack: pd.DataFrame,
) -> pd.DataFrame:
    return bccwj_eyetrack[bccwj_eyetrack["num_unk"] == 0]


def main(args):
    bccwj_eyetrack = pd.read_csv(args.input_path, sep=",")
    bccwj_eyetrack = _add_prev_info(bccwj_eyetrack, "length")
    bccwj_eyetrack = _add_prev_info(bccwj_eyetrack, "count_ave_kika")
    bccwj_eyetrack = _remove_out_of_main_text(bccwj_eyetrack)
    bccwj_eyetrack = _remove_not_fixated(bccwj_eyetrack)
    bccwj_eyetrack = _remove_unk(bccwj_eyetrack)
    bccwj_eyetrack.to_csv(args.save_path, sep=",", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
