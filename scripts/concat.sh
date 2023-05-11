mkdir -p data/aggregated/concat
python src/aggregate/concat.py \
    --bccwj_eyetrack_path data/BCCWJ-EyeTrack/fpt.csv \
    --lstm_root data/aggregated/lstm \
    --lstm_seeds 1111 1112 1113 \
    --rnng_root data/aggregated/rnng \
    --rnng_strategies top_down in_order \
    --rnng_seeds  3435 3436 3437 \
    --rnng_beams 100 200 400 600 800 1000 \
    --unk_root data/aggregated/unk \
    --units A B C D \
    --save_path data/aggregated/concat/fpt.csv
