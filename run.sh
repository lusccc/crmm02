# @@@@@@ cr
# pretrain
# rolling window split
python main_runner.py --task pretrain --per_device_train_batch_size 230 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr --dataset_split_strategy "rolling_window" --num_train_samples 200 --train_years "2010,2011,2012" --test_years "2013" --use_val --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --contrastive_targets "num,text"
# random split
python main_runner.py --task pretrain --per_device_train_batch_size 230 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr --dataset_split_strategy "random" --num_train_samples 200 --use_val --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --contrastive_targets "num,text"
# pair match prediction
python main_runner.py --task clip_pair_match_prediction --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-27_22-22-56_hMe/output" --save_excel_path "./excel/res_1128.xlsx" --natural_language_labels "poor@good"
# finetune
python main_runner.py --task finetune_for_classification --per_device_train_batch_size 240 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-28_21-58-04_tae/output" --save_excel_path "./excel/res_1128_finetune.xlsx"
# finetune scratch
python main_runner.py --task finetune_for_classification_from_scratch --per_device_train_batch_size 350 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_scratch.xlsx"
# multi task classification
python main_runner.py --task clip_multi_task_classification --per_device_train_batch_size 345 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --dataset_split_strategy "rolling_window" --train_years "2010,2011,2012" --test_years "2013" --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_cr_2013_cat_sent.xlsx"
# no clip classification
python main_runner.py --task no_clip_classification --per_device_train_batch_size 85 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"
#


# @@@@@@ cr2
# train
python main_runner.py --task pretrain --per_device_train_batch_size 150 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --contrastive_targets "num,text"
# test
python main_runner.py --task clip_pair_match_prediction --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method conv  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-15_23-26-57_kxp/output" --save_excel_path "./excel/res_roberta.xlsx" --natural_language_labels "weak[SEP]poor[SEP]challenges[SEP]concerning@good[SEP]strong[SEP]favorable[SEP]healthy"
# finetune
python main_runner.py --task finetune_for_classification --per_device_train_batch_size 400 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-27_22-02-23_eQm/output" --save_excel_path "./excel/res_1128.xlsx"
# only finetune eval
python main_runner.py --task finetune_for_classification --per_device_train_batch_size 1000 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method conv  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-22_13-44-22_RNh/output" --save_excel_path "./excel/res_finetune_prompt_concat.xlsx"
# finetune scratch
python main_runner.py --task finetune_for_classification_from_scratch --per_device_train_batch_size 350 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_scratch_repeat"
# multi task classification
python main_runner.py --task clip_multi_task_classification --per_device_train_batch_size 36 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"
# no clip classification
python main_runner.py --task no_clip_classification --per_device_train_batch_size 400 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"


# @@@@@@ benchmark
python benchmark_model_comparison.py --dataset_name cr --data_path "./data/cr" --excel_path "res_benchmark_cr_2013.xlsx" --dataset_split_strategy "rolling_window" --train_years 2010,2011,2012 --test_years 2013 --cat_encoder "onehot"

python benchmark_model_comparison.py --dataset_name cr2 --data_path "./data/cr2" --excel_path "res_benchmark_random.xlsx" --dataset_split_strategy "random"
