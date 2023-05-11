import argparse
import pandas as pd

from aggregate_surp_lstm import load_surp_lstm
from utils import filter_bccwj_eyetrack, normal_bracket2ptb_bracket, zen2han

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bccwj_eyetrack_path", type=str, default="data/BCCWJ-EyeTrack/fpt.csv"
)
parser.add_argument("--unit", type=str)
parser.add_argument("--lstm_surp_path", type=str)
parser.add_argument("--test_path", type=str)
parser.add_argument("--save_path", type=str)


def aggregate_num_unk(
    bccwj_eyetrack: pd.DataFrame,
    surp_lstm: pd.DataFrame,
) -> pd.DataFrame:
    aggregated_num_unks = []
    aggregated_surfaces = []
    surface_id = 0
    for i in range(len(bccwj_eyetrack["surface"])):
        target_surface = zen2han(
            normal_bracket2ptb_bracket(bccwj_eyetrack["surface"][i])
        )

        aggregated_num_unk = 0
        aggregated_surface = ""

        for j in range(surface_id, len(surp_lstm)):
            if surp_lstm["surface"][j] == "▁":
                aggregated_num_unks[-1] += 1
                surface_id += 1
            else:
                if surp_lstm["word"][j] == "<unk>":
                    aggregated_num_unk += 1
                surface_id += 1
                aggregated_surface += zen2han(surp_lstm["surface"][j]).replace("▁", "")

            if aggregated_surface == target_surface:
                aggregated_num_unks.append(aggregated_num_unk)
                aggregated_surfaces.append(aggregated_surface)
                break
        assert (
            len(aggregated_num_unks) == i + 1
        ), "aggregated_num_unk is not correct at {}th surface.".format(i)

    return pd.DataFrame(
        {
            "surface": aggregated_surfaces,
            "num_unk": aggregated_num_unks,
        }
    )


def main(args):
    bccwj_eyetrack = pd.read_csv(args.bccwj_eyetrack_path, sep=",")
    filtered_bccwj_eyetrack = filter_bccwj_eyetrack(args.unit, bccwj_eyetrack)
    surp_lstm = load_surp_lstm(args.lstm_surp_path, args.test_path)

    num_unk = aggregate_num_unk(filtered_bccwj_eyetrack, surp_lstm)
    num_unk.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
