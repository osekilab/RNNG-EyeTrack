mkdir -p neural-complexity/outputs;
for seed in 1111 1112 1113;
    do
    for unit in "A" "B" "C" "D";
        do
            python neural-complexity/main.py \
            --model_file neural-complexity/models/${seed}.pt \
            --vocab_file neural-complexity/data/npcmj-vocab.txt \
            --data_dir data/bccwj \
            --testfname ${unit}.lstm.txt \
            --cuda \
            --test \
            --words \
            --nopp > neural-complexity/outputs/surprisals_${seed}_${unit}.txt
        done
    done
