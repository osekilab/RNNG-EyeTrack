# Modeling Human Sentence Processing with Left-Corner Recurrent Neural Network Grammars

This repository provides the code for the paper [Modeling Human Sentence Processing with Left-Corner Recurrent Neural Network Grammars](https://aclanthology.org/2021.emnlp-main.235/).

> [Modeling Human Sentence Processing with Left-Corner Recurrent Neural Network Grammars](https://aclanthology.org/2021.emnlp-main.235/) <br>
> Ryo Yoshida, Hiroshi Noji, and Yohei Oseki <br>
> EMNLP 2021

## Requirements
- `python==3.9.13`
- `R==4.2.2`

## Installation
```bash
git clone git@github.com:osekilab/RNNG-LC.git
cd RNNG-LC
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Training data preparation
1. Download the NPCMJ corpus from [here](https://npcmj.ninjal.ac.jp/index.html).
   - A version from 2021-03-02 was used in the paper. This version is not currently available on the website; if you want to download this version, please contact the authors for the data.
2. Unzip and place the data on `data/`.
3. Run the following command to preprocess the data:
   ```bash
   python src/parser-data-gen/preprocess.py \
      --file_root data/npcmj \
      --save_root data/npcmj_preprocessed \
      --error_root data/npcmj_error
   ```
4. Run the following command to split train/dev/test data:
   ```bash
   python src/parser-data-gen/split.py \
      --preprocessed_root data/npcmj_preprocessed \
      --save_root data/npcmj_split
   ```

## Training
If you want to download the models trained in the paper, please contact the authors.

### RNNG
1. Install [rnng-pytorch](https://github.com/aistairc/rnng-pytorch).
   ```bash
   git clone https://github.com/aistairc/rnng-pytorch.git
   cd rnng-pytorch
   git checkout f9a5663
   ```
   - Version `f9a5663` was used in the paper.
2. Run the following command to preprocess the data for RNNGs:
   ```bash
   python preprocess.py \
      --vocabsize 8000 \
      --unkmethod subword \
      --subword_type bpe \
      --trainfile ../data/npcmj_split/train.mrg \
      --valfile ../data/npcmj_split/dev.mrg \
      --testfile ../data/npcmj_split/test.mrg \
      --outputfile ./data/npcmj \
      --keep_ptb_bracket
   ```
3. Train RNNGs.
   ```bash
   mkdir models
   cd ..
   bash scripts/train_rnng.sh
   ```

### LSTM
1. Install [neural-complexity](https://github.com/vansky/neural-complexity).
   ```bash
   git clone https://github.com/vansky/neural-complexity.git
   cd neural-complexity
   git checkout tags/v1.1.0
   ```
   - Version `v1.1.0` was used in the paper.
2. Run the following command to preprocess the training data for LSTMs:
   ```bash
   python src/lstm-data-gen/train_data_gen.py \
      --rnng_data_root rnng-pytorch/data/ \
      --train_file npcmj-train.json \
      --val_file npcmj-val.json \
      --test_file npcmj-test.json
   ```
3. Train LSTMs.
   ```bash
   mkdir models
   cd ..
   bash scripts/train_lstm.sh
   ```

## Evaluation data preparation
   We cannot share the original text of the evaluation data due to copyright issues; please contact [https://github.com/masayu-a/BCCWJ-EyeTrack](https://github.com/masayu-a/BCCWJ-EyeTrack). <br>
   After obtaining the data,
   1. tokenize each sentence in the original text with [HARUNIWA2](http://www.compling.jp/ajb129/haruniwa2.html)
   2. and replace round brackets with PTB-style brackets.

   Then, place the text data of each unit (`[A-D].txt`) on `data/bccwj/`.
   - NOTE: We manually corrected some tokenization errors, in which the boundaries of phrasal units were not split.

## Surprisal calculation
### RNNG
1. Calculate surprisals with RNNGs.
   ```bash
   bash scripts/calc_surp_rnng.sh
   ```

### LSTM
1. Run the following command to preprocess the evaluation data for LSTMs:
   ```bash
   python src/lstm-data-gen/eval_data_gen.py \
      --eval_data_root data/bccwj \
      --A_file A.txt \
      --B_file B.txt \
      --C_file C.txt \
      --D_file D.txt \
      --spm_model rnng-pytorch/data/npcmj-spm.model
   ```

2. Calculate surprisals with LSTMs.
   ```bash
   bash scripts/calc_surp_lstm.sh
   ```

## Aggregation
### Eye-tracking data preparation
1. Download the BCCWJ-EyeTrack from [here](https://github.com/masayu-a/BCCWJ-EyeTrack).
2. Add the original text of each phrasal unit to the `surface` column in `fpt.csv`.
3. Place the `fpt.csv` file on `data/BCCWJ-EyeTrack/`.

### RNNG
1. Aggregate RNNGs surprisals.
   ```bash
   bash scripts/aggregate_surp_rnng.sh
   ```

### LSTM
1. Aggregate LSTMs surprisals.
   ```bash
   bash scripts/aggregate_surp_lstm.sh
   ```

### UNK
1. Aggregate the number of unknown words.
   ```bash
   bash scripts/aggregate_num_unk.sh
   ```
   - NOTE: The number of unknown words may vary depending on the version of the evaluation data.

### Concat
1. Concatenate the above data.
   ```bash
   bash scripts/concat.sh
   ```

### Post-processing
1. Run the following command to post-process the data:
   ```bash
   python src/aggregate/post_process.py \
      --input_path data/aggregated/concat/fpt.csv \
      --save_path data/aggregated/concat/fpt-del.csv
   ```

## Evaluation
Script for the evaluation is in `r-workspace/modeling.R`. <br>
We recommend using [RStudio](https://rstudio.com/) to run the script.

## License
MIT

## Note
If you want to download the 2021-03-02 version of the NPCMJ and the models trained in our paper, please contact `yoshiryo0617 [at] g.ecc.u-tokyo.ac.jp`
