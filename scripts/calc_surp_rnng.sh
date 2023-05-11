mkdir -p rnng-pytorch/outputs;
for strategy in "top_down" "in_order";
    do
    for seed in 3435 3436 3437;
        do
        for beam in 100 200 400 600 800 1000;
            do
            for unit in "A" "B" "C" "D";
                do
                    word_beam=`expr ${beam} \/ 10`
                    fast_track=`expr ${beam} \/ 100`
                    python rnng-pytorch/beam_search.py \
                    --test_file data/bccwj/${unit}.txt \
                    --lm_output_file rnng-pytorch/outputs/surprisals_${strategy}_${seed}_${beam}_${unit}.txt \
                    --model_file rnng-pytorch/models/${strategy}_${seed}.pt \
                    --beam_size ${beam} \
                    --word_beam_size ${word_beam} \
                    --shift_size ${fast_track} > rnng-pytorch/outputs/trees_${strategy}_${seed}_${beam}_${unit}.txt
                done
            done
        done
    done

