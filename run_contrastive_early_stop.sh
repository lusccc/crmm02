#!/bin/bash

# Define variables
dataset="cr"
patience=5
root_dir="./exps"
data_path="./data/cr"
language_model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

# Ensure the excel directory exists
mkdir -p "./excel"

# Loop through the specified ranges
for r in 1 2 3 4; do
  repeat=$r
  for i in 10 20 30 40 50 60 70 80 90; do
    contrastive_early_stopping_epoch=$i
    excel_file="./excel/${dataset}_res_random_split_multitask_epoch_100_gelu_noCatSent_catLM_unfreezeEmbPoolerAddNorm_earlyStopContrastive_${contrastive_early_stopping_epoch}_patience_${patience}_on_valLoss_rep${repeat}.xlsx"

    # Run Python script with the specified parameters
    python main_runner.py \
      --root_dir "$root_dir" \
      --task "clip_multi_task_classification" \
      --per_device_train_batch_size 40 \
      --num_train_epochs 100 \
      --data_path "$data_path" \
      --dataset_name "$dataset" \
      --dataset_split_strategy "random" \
      --modality_fusion_method "conv" \
      --use_modality num,cat,text \
      --fuse_modality num,cat \
      --language_model_name "$language_model" \
      --load_hf_model_from_cache True \
      --save_excel_path "$excel_file" \
      --patience $patience \
      --freeze_language_model_params \
      --contrastive_early_stopping_epoch $contrastive_early_stopping_epoch

    # Check if the python command exited with a non-zero status
    if [ $? -ne 0 ]; then
      echo "Python script returned an error on iteration $repeat with early stopping epoch $contrastive_early_stopping_epoch"
      # Exit the loop or handle the error as needed
      exit 1
    fi
  done
done