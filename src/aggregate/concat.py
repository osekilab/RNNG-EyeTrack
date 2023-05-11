import argparse

import pandas as pd

from utils import normal_bracket2ptb_bracket, zen2han

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bccwj_eyetrack_path", type=str, default="data/BCCWJ-EyeTrack/fpt.csv"
)

parser.add_argument("--lstm_root", type=str, default="data/aggregated/lstm")
parser.add_argument("--lstm_seeds", type=int, nargs="+", default=[1111, 1112, 1113])

parser.add_argument("--rnng_root", type=str, default="data/aggregated/rnng")
parser.add_argument(
    "--rnng_strategies", type=str, nargs="+", default=["top_down", "in_order"]
)
parser.add_argument("--rnng_seeds", type=int, nargs="+", default=[3435, 3436, 3437])
parser.add_argument(
    "--rnng_beams", type=int, nargs="+", default=[100, 200, 400, 600, 800, 1000]
)

parser.add_argument("--unk_root", type=str, default="data/aggregated/unk")

parser.add_argument("--units", type=str, nargs="+", default=["A", "B", "C", "D"])

parser.add_argument("--save_path", type=str, default="data/aggregated/concat/fpt.csv")


def main(args):
    bccwj_eyetrack = pd.read_csv(args.bccwj_eyetrack_path, sep=",")
    surface = (
        bccwj_eyetrack["surface"]
        .apply(normal_bracket2ptb_bracket)
        .apply(zen2han)
        .values
    )

    concat_dict = {}
    # LSTM
    for seed in args.lstm_seeds:
        lstm_dfs = []
        for unit in args.units:
            lstm_path = f"{args.lstm_root}/{seed}_{unit}.csv"
            lstm_df = pd.read_csv(lstm_path, sep=",")
            lstm_dfs += [lstm_df] * 12
        lstm_dfs = pd.concat(lstm_dfs).reset_index(drop=True)
        assert (surface == lstm_dfs["surface"].values).all()
        concat_dict[f"lstm_{seed}"] = lstm_dfs["surps"]

    # RNNG
    for strategy in args.rnng_strategies:
        for seed in args.rnng_seeds:
            for beam in args.rnng_beams:
                rnng_dfs = []
                for unit in args.units:
                    rnng_path = f"{args.rnng_root}/{strategy}_{seed}_{beam}_{unit}.csv"
                    rnng_df = pd.read_csv(rnng_path, sep=",")
                    rnng_dfs += [rnng_df] * 12
                rnng_dfs = pd.concat(rnng_dfs).reset_index(drop=True)
                assert (surface == rnng_dfs["surface"].values).all()
                concat_dict[f"rnng_{strategy}_{seed}_{beam}"] = rnng_dfs["surps"]
    # UNK
    unk_dfs = []
    for unit in args.units:
        unk_path = f"{args.unk_root}/{unit}.csv"
        unk_df = pd.read_csv(unk_path, sep=",")
        unk_dfs += [unk_df] * 12
    unk_df = pd.concat(unk_dfs).reset_index(drop=True)
    assert (surface == unk_df["surface"].values).all()
    concat_dict["num_unk"] = unk_df["num_unk"]

    concat_df = pd.DataFrame(concat_dict)
    bccwj_eyetrack = pd.concat([bccwj_eyetrack, concat_df], axis=1)
    bccwj_eyetrack.to_csv(args.save_path, sep=",", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
