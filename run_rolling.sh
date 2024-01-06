#!/bin/bash
set -e # 当任何命令返回非零退出状态时，脚本会立即退出

run_experiment() {
    local train_years="$1"
    local test_years="$2"
    local repeat="$3"

    python main_runner.py \
        --root_dir ./exps \
        --task "$task" \
        --per_device_train_batch_size "$batch_size" \
        --per_device_eval_batch_size 40 \
        --num_train_epochs "$epoch" \
        --data_path "./data/$dataset" \
        --dataset_name "$dataset" \
        --dataset_split_strategy "$dataset_split_strategy" \
        --train_years "$train_years" \
        --test_years "$test_years" \
        --freeze_language_model_params True \
        --use_modality ${use_modality} \
        --fuse_modality "num,cat" \
        --contrastive_targets ${contrastive_targets} \
        --clncp_ensemble_method ${ensemble} \
        --num_cat_language_model_hyperparameters ${hp} \
        --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" \
        --load_hf_model_from_cache True \
        --save_excel_path "./excel/${dataset}_${task_short}_${dataset_split_strategy}_#${train_years}#_#${test_years}#_${exp_desc}.xlsx" \
        --save_hist_eval_csv_path "./hist_csv/hist_${dataset}_${task_short}_${dataset_split_strategy}_#${train_years}#_#${test_years}#_${repeat}_${exp_desc}.csv"
    
    # 检查Python命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Python script failed with status $?"
        exit 1
    fi
}

# Main loop
main_loop(){
    for (( i=${start_i}; i<repeat; i++ )); do
        for index in "${!train_years_list[@]}"; do
            run_experiment "${train_years_list[$index]}" "${test_year_list[$index]}" "${i}"
        done
    done
}


# ******************************************
start_i=6
#start_i=1
repeat=5+6
dataset_split_strategy="rolling_window"
train_years_list=(
  "2010,2011,2012"
  "2011,2012,2013"
  "2012,2013,2014"
  "2013,2014,2015"
)
test_year_list=(
  "2013"
  "2014"
  "2015"
  "2016"
)
#default_hp="512,8,8,2048"
# @@@@ 1.
dataset="cr"
task="finetune_classification_scratch"
task_short="scratch"
use_modality="num,cat"
contrastive_targets="None"
batch_size=1000
epoch=100
hp="512,8,8,2048"
ensemble="weighted_avg"
exp_desc="numLayernormIndpFeatLayerEmb_jointDrp.2AddClsToken_bertcnf${hp}-vocabsize_fixWrongPadId_clncp_${ensemble}_epoch${epoch}"
main_loop


#########################################################################################################################

## @@@@ 1.
#dataset="cr"
#task="multi_task_classification"
#task_short="multitask"
#use_modality="num,cat,text"
#contrastive_targets="joint,text"
#batch_size=1000
#epoch=40
#hp="256,4,4,1024"
#ensemble="weighted_avg"
#exp_desc="numLayernormIndpFeatLayerEmb_jointDrp.2AddClsToken_bertcnf${hp}-vocabsize_fixWrongPadId_clncp_${ensemble}"
#main_loop
#
## @@@@ 2.
#dataset="cr"
#task="finetune_classification_scratch"
#task_short="scratch"
#use_modality="num,cat"
#contrastive_targets="None"
#batch_size=1000
#epoch=40
#hp="256,4,4,1024"
#ensemble="no_ensemble"
#exp_desc="numLayernormIndpFeatLayerEmb_jointDrp.2AddClsToken_bertcnf${hp}-vocabsize_fixWrongPadId_clncp_${ensemble}"
#main_loop
#
## @@@@ 3.
#dataset="cr2"
#task="multi_task_classification"
#task_short="multitask"
#use_modality="num,cat,text"
#contrastive_targets="joint,text"
#batch_size=1000
#epoch=50
#hp="256,4,4,1024"
#ensemble="weighted_avg"
#exp_desc="numLayernormIndpFeatLayerEmb_jointDrp.2AddClsToken_bertcnf${hp}-vocabsize_fixWrongPadId_clncp_${ensemble}"
#main_loop
#
## @@@@ 4.
#dataset="cr2"
#task="finetune_classification_scratch"
#task_short="scratch"
#use_modality="num,cat"
#contrastive_targets="None"
#batch_size=1000
#epoch=50
#hp="256,4,4,1024"
#ensemble="no_ensemble"
#exp_desc="numLayernormIndpFeatLayerEmb_jointDrp.2AddClsToken_bertcnf${hp}-vocabsize_fixWrongPadId_clncp_${ensemble}"
#main_loop