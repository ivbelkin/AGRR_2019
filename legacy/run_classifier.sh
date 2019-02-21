#! /bin/bash

DATA_DIR=./orgs
BASE_LOG_DIR=~/nvme/agrr_logs

python run_classifier.py \
    --data_dir $DATA_DIR \
    --bert_model bert-base-multilingual-cased \
    --task_name AGRR \
    --output_dir $BASE_LOG_DIR/cls_finetune_baseline \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --train_batch_size 32 \
    --save_n_best 5 \
    --learning_rate 4e-5 \
    --num_train_epochs 3 \
    --n_cycles 10
