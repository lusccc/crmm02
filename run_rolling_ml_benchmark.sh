#!/bin/bash
set -e # 当任何命令返回非零退出状态时，脚本会立即退出

dataset="cr"
dataset_split_strategy="rolling_window"
repeat=10

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


run_experiment() {
    local train_years="$1"
    local test_years="$2"
    local repeat="$3"

    python ml_benchmark_model_comparison.py \
        --data_path "./data/$dataset" \
        --dataset_name "$dataset" \
        --dataset_split_strategy "$dataset_split_strategy" \
        --train_years "$train_years" \
        --test_years "$test_years" \
        --excel_path "./excel/res_mlbenchmark_${dataset}_${dataset_split_strategy}_#${train_years}#_#${test_years}#_rep${repeat}.xlsx"

    # 检查Python命令的退出状态
    if [ $? -ne 0 ]; then
        echo "Python script failed with status $?"
        exit 1
    fi
}
main_loop() {
    # Main loop
    for (( i=0; i<repeat; i++ )); do
        for index in "${!train_years_list[@]}"; do
            run_experiment "${train_years_list[$index]}" "${test_year_list[$index]}" "${i}"
        done
    done
}

main_loop

dataset="cr2"

main_loop