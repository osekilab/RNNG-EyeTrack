import argparse
import pandas as pd

from utils import filter_bccwj_eyetrack, normal_bracket2ptb_bracket, zen2han


parser = argparse.ArgumentParser()
parser.add_argument(
    "--bccwj_eyetrack_path", type=str, default="data/BCCWJ-EyeTrack/fpt.csv"
)
parser.add_argument("--unit", type=str)
parser.add_argument("--rnng_surp_path", type=str)
parser.add_argument("--save_path", type=str)


def load_surp_rnng(rnng_surp_path: str) -> pd.DataFrame:
    df = pd.read_csv(rnng_surp_path, sep="\t", header=None)
    df.columns = [
        "sent_id",
        "sent_pos",
        "surface",
        "unkified_surface",
        "surp",
        "piece_surp",
    ]
    df = df[:-2]
    return df


def aggregate_surp_rnng(
    bccwj_eyetrack: pd.DataFrame,
    surp_rnng: pd.DataFrame,
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

        for j in range(surface_id, len(surp_rnng)):
            aggregated_surp += surp_rnng["surp"][j]
            aggregated_surface += zen2han(surp_rnng["surface"][j])
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
    surp_rnng = load_surp_rnng(args.rnng_surp_path)
    aggregated_surp_rnng = aggregate_surp_rnng(
        filtered_bccwj_eyetrack,
        surp_rnng,
    )
    aggregated_surp_rnng.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
