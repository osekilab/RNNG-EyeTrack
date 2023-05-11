import mojimoji
import pandas as pd


def zen2han(text: str) -> str:
    return mojimoji.zen_to_han(text, kana=False)


def normal_bracket2ptb_bracket(text: str) -> str:
    return text.replace("（", "-LRB-").replace("）", "-RRB-")


def filter_bccwj_eyetrack(unit: str, bccwj_eyetrack: pd.DataFrame) -> pd.DataFrame:
    unit2subj = {
        "A": 1,
        "B": 2,
        "C": 5,
        "D": 7,
    }
    unit2sample = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
    }
    return bccwj_eyetrack[
        (bccwj_eyetrack["subj"] == unit2subj[unit])
        & (bccwj_eyetrack["sample"] == unit2sample[unit])
    ].reset_index(drop=True)
