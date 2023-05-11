python -m nltk.downloader punkt

for seed in 1111 1112 1113;
    do
        python neural-complexity/main.py \
            --emsize 256 \
            --nhid 256 \
            --epochs 40 \
            --batch_size 64 \
            --lr 20 \
            --dropout 0.2 \
            --model_file neural-complexity/models/${seed}.pt \
            --vocab_file neural-complexity/data/npcmj-vocab.txt \
            --tied \
            --cuda \
            --data_dir neural-complexity/data \
            --trainfname npcmj-train.txt \
            --validfname npcmj-val.txt \
            --seed ${seed} \
            > neural-complexity/models/${seed}.pt.log
    done
