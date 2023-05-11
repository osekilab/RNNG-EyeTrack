mkdir -p data/aggregated/lstm
for seed in 1111 1112 1113;
    do
    for unit in "A" "B" "C" "D";
        do
        python src/aggregate/aggregate_surp_lstm.py \
            --bccwj_eyetrack_path data/BCCWJ-EyeTrack/fpt.csv \
            --unit ${unit} \
            --lstm_surp_path neural-complexity/outputs/surprisals_${seed}_${unit}.txt \
            --test_path data/bccwj/${unit}.lstm.txt \
            --save_path data/aggregated/lstm/${seed}_${unit}.csv
        done
    done
