# @@@@@@ cr
# pretrain
# (1)contrastive_targets :
python main_runner.py --task pretrain --per_device_train_batch_size 230 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --contrastive_targets "num,text"
# (2)contrastive_targets :
python main_runner.py --task pretrain --per_device_train_batch_size 110 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --contrastive_targets "joint,text" --pretrained_model_dir "./exps/pretrain_2023-11-27_11-59-06_EcJ/output"

# pair match prediction
python main_runner.py --task clip_pair_match_prediction --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params True  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-27_22-22-56_hMe/output" --save_excel_path "./excel/res_1128.xlsx" --natural_language_labels "poor@good"
# finetune
python main_runner.py --task finetune_for_classification --per_device_train_batch_size 240 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-28_21-58-04_tae/output" --save_excel_path "./excel/res_1128_finetune.xlsx"
# finetune scratch
python main_runner.py --task finetune_for_classification_from_scratch --per_device_train_batch_size 350 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_scratch.xlsx"
# multi task classification
python main_runner.py --task clip_multi_task_classification --per_device_train_batch_size 18 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_1127.xlsx" --pretrained_model_dir "./exps/pretrain_2023-11-27_11-59-06_EcJ/output"
# no clip classification
python main_runner.py --task no_clip_classification --per_device_train_batch_size 85 --num_train_epochs 100 --data_path ./data/cr --dataset_name cr  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"


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
python main_runner.py --task finetune_for_classification_from_scratch --per_device_train_batch_size 350 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "prompt_concat"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res.xlsx"
# multi task classification
python main_runner.py --task clip_multi_task_classification --per_device_train_batch_size 36 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"
# no clip classification
python main_runner.py --task no_clip_classification --per_device_train_batch_size 400 --num_train_epochs 100 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method "conv"  --freeze_language_model_params False  --use_modality num,cat,text --fuse_modality num,cat --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True  --save_excel_path "./excel/res_clip_multi_task.xlsx"


# @@@@@@ benchmark
python benchmark_model_comparison.py --data_path data/cr --excel_path ./excel/res_bm_cr.xlsx
python benchmark_model_comparison.py --data_path data/cr2 --excel_path ./excel/res_bm_cr2.xlsx