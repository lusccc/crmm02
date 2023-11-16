# @@@@@@ cr
# train
python main_runner.py --task pretrain --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr --dataset_name cr  --modality_fusion_method conv  --freeze_text_params True  --use_modality num,cat,text --fuse_modality num,cat --text_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True
# test
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr --dataset_name cr  --modality_fusion_method conv  --freeze_text_params True  --use_modality num,cat,text --fuse_modality num,cat --text_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-15_23-21-44_FBo/output" --save_excel_path "./excel/res_roberta.xlsx" --natural_language_labels "bad poor@good"


# @@@@@@ cr2
# train
python main_runner.py --task pretrain --per_device_train_batch_size 150 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method conv  --freeze_text_params True  --use_modality num,cat,text --fuse_modality num,cat  --text_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True
# test
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 120 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2  --modality_fusion_method conv  --freeze_text_params True  --use_modality num,cat,text --fuse_modality num,cat --text_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --pretrained_model_dir "./exps/pretrain_2023-11-15_23-26-57_kxp/output" --save_excel_path "./excel/res_roberta.xlsx"



# @@@@@@ benchmark
python benchmark_model_comparison.py --data_path data/cr --excel_path ./excel/res_bm_cr.xlsx
python benchmark_model_comparison.py --data_path data/cr2 --excel_path ./excel/res_bm_cr2.xlsx