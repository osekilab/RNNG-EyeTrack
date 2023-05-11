mkdir -p data/aggregated/unk;
for unit in "A" "B" "C" "D";
    do
    python src/aggregate/aggregate_num_unk.py \
        --bccwj_eyetrack_path data/BCCWJ-EyeTrack/fpt.csv \
        --unit ${unit} \
        --lstm_surp_path neural-complexity/outputs/surprisals_1111_${unit}.txt \
        --test_path data/bccwj/${unit}.lstm.txt \
        --save_path data/aggregated/unk/${unit}.csv
    done
