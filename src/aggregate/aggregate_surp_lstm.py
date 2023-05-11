import argparse
import math
import pandas as pd
from typing import List

from utils import filter_bccwj_eyetrack, normal_bracket2ptb_bracket, zen2han

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bccwj_eyetrack_path", type=str, default="data/BCCWJ-EyeTrack/fpt.csv"
)
parser.add_argument("--unit", type=str)
parser.add_argument("--lstm_surp_path", type=str)
parser.add_argument("--test_path", type=str)
parser.add_argument("--save_path", type=str)


def log2e(x: float) -> float:
    return math.log(2**x)


def _load_surface(test_path: str) -> List[str]:
    with open(test_path, "r") as f:
        surface = []
        for line in f:
            surface += line.rstrip().split(" ")
    return surface


def load_surp_lstm(lstm_surp_path: str, test_path: str) -> pd.DataFrame:
    df = pd.read_csv(lstm_surp_path, sep=" ")
    surface = _load_surface(test_path)

    assert len(df) == len(surface)

    df["surface"] = surface
    df["surp"] = df["surp"].apply(log2e)
    return df


def aggregate_surp_lstm(
    bccwj_eyetrack: pd.DataFrame,
    surp_lstm: pd.DataFrame,
) -> pd.DataFrame:
    aggregated_surps = []
    aggregated_surfaces = []
    surface_id = 0
    for i in range(len(bccwj_eyetrack["surface"])):
        target_surface = zen2han(
            normal_bracket2ptb_bracket(bccwj_eyetrack["surface"][i])
        )

        aggregated_surp = 0.0
        aggregated_surface = ""

        for j in range(surface_id, len(surp_lstm)):
            if surp_lstm["surface"][j] == "▁":
                aggregated_surps[-1] += surp_lstm["surp"][j]
                surface_id += 1
            else:
                aggregated_surp += surp_lstm["surp"][j]
                aggregated_surface += zen2han(surp_lstm["surface"][j]).replace("▁", "")
                surface_id += 1

            if aggregated_surface == target_surface:
                aggregated_surps.append(aggregated_surp)
                aggregated_surfaces.append(aggregated_surface)
                break
        assert (
            len(aggregated_surps) == i + 1
        ), "aggregated_surp is not correct at {}th surface.".format(i)

    return pd.DataFrame(
        {
            "surface": aggregated_surfaces,
            "surps": aggregated_surps,
        }
    )


def main(args):
    bccwj_eyetrack = pd.read_csv(args.bccwj_eyetrack_path, sep=",")
    filtered_bccwj_eyetrack = filter_bccwj_eyetrack(
        args.unit,
        bccwj_eyetrack,
    )
    surp_lstm = load_surp_lstm(args.lstm_surp_path, args.test_path)
    aggregated_surp_lstm = aggregate_surp_lstm(
        filtered_bccwj_eyetrack,
        surp_lstm,
    )
    aggregated_surp_lstm.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
