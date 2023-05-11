for strategy in "top_down" "in_order";
    do
    for seed in 3435 3436 3437;
        do
            python rnng-pytorch/train.py \
                --train_file rnng-pytorch/data/npcmj-train.json \
                --val_file rnng-pytorch/data/npcmj-val.json \
                --sp_model rnng-pytorch/data/npcmj-spm.model \
                --fixed_stack \
                --strategy ${strategy} \
                --w_dim 256 \
                --h_dim 256 \
                --dropout 0.3 \
                --optimizer 'adam' \
                --batch_size 64 \
                --lr 0.001 \
                --num_epochs 40 \
                --save_path rnng-pytorch/models/${strategy}_${seed}.pt \
                --tensorboard_log_dir rnng-pytorch/tensorboards \
                --batch_group similar_action_length \
                --seed ${seed} \
                --device 'cpu'
        done
    done
