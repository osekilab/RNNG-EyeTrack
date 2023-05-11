mkdir -p data/aggregated/rnng;
for strategy in "top_down" "in_order";
    do
    for seed in 3435 3436 3437;
        do
        for beam in 100 200 400 600 800 1000;
            do
            for unit in "A" "B" "C" "D";
                do
                python src/aggregate/aggregate_surp_rnng.py \
                    --bccwj_eyetrack_path data/BCCWJ-EyeTrack/fpt.csv \
                    --unit ${unit} \
                    --rnng_surp_path rnng-pytorch/outputs/surprisals_${strategy}_${seed}_${beam}_${unit}.txt \
                    --save_path data/aggregated/rnng/${strategy}_${seed}_${beam}_${unit}.csv
                done
            done
        done
    done
