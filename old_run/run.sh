
# eval 1030
# cr:     ./exps/pretrain_2023-10-29_21-03-05_din/output       AutogluonModels/ag-20231105_051122
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 250 --data_path ./data/cr --dataset_name cr --label_col "binaryRating" --feature_transform_res_dir AutogluonModels/ag-20231105_051122  --pretrained_model_dir ./exps/pretrain_2023-11-05_23-10-28_2QM/output  --modality_fusion_method conv  --fuse_modality num,cat

# cr2:    ./exps/pretrain_2023-10-30_16-12-24_din/output
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 250 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating" --feature_transform_res_dir AutogluonModels/ag-20231030_092803  --pretrained_model_dir exps/pretrain_2023-11-04_20-36-32_0Zv/output --save_excel_path ./excel/res_multi_clip_loss.xlsx

#train -------------
python main_runner.py --task conventional_multimodal_classification --per_device_train_batch_size 250 --num_train_epochs 100  --data_path ./data/cr --dataset_name cr --label_col "binaryRating"   --modality_fusion_method conv --fuse_modality num,cat --patience 20 --save_excel_path ./excel/res_conventional_classification.xlsx


python main_runner.py --task pretrain --per_device_train_batch_size 18 --num_train_epochs 300 --data_path ./data/cr --dataset_name cr  --modality_fusion_method conv  --freeze_text_params False  --use_modality num,cat,text --fuse_modality num,cat --feature_transform_res_dir AutogluonModels/ag-20231105_051122

python main_runner.py --task pretrain --per_device_train_batch_size 27 --num_train_epochs 300 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating" --text_cols GPT_description --modality_fusion_method conv  --freeze_clip_text_params False  --use_modality num,cat,text --fuse_modality num,cat  --natural_language_labels 'Poor credit@Good credit'








python main_runner.py --task conventional_multimodal_classification --per_device_train_batch_size 250 --num_train_epochs 100 --feature_transform_res_dir AutogluonModels/ag-20231030_092803 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating"  --modality_fusion_method conv --fuse_modality num,cat --patience 10 --save_excel_path ./excel/res_conventional_classification.xlsx


python main_runner.py --task pretrain --per_device_train_batch_size 250 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating"  --modality_fusion_method concat




python main_runner.py --task pretrain --per_device_train_batch_size 300 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating"  --modality_fusion_method conv   --freeze_clip_text_params False --num_train_epochs 25

python main_runner.py --task pretrain --per_device_train_batch_size 275 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating"  --modality_fusion_method conv   --freeze_clip_text_params False --num_train_epochs 100 --feature_transform_res_dir AutogluonModels/ag-20231030_092803 --load_best_model_at_end False

# zero shot test
python main_runner.py --task zero_shot_prediction_from_another_dataset_pretraining  --data_path ./data/cr --dataset_name cr --label_col "binaryRating" --feature_transform_res_dir AutogluonModels/ag-20231026_024542  --pretrained_model_dir exps/pretrain_2023-10-31_21-31-44_d62/output --save_excel_path ./excel/res_zeroshot.xlsx

# benchmark
python benchmark_model_comparison.py --data_path data/cr --excel_path ./excel/res_bm_cr.xlsx
python benchmark_model_comparison.py --data_path data/cr2 --excel_path ./excel/res_bm_cr2.xlsx