
# eval 1030
# cr: AutogluonModels/ag-20231026_024542    ./exps/pretrain_2023-10-29_21-03-05_din/output
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 250 --data_path ./data/cr --dataset_name cr --label_col "binaryRating" --feature_transform_res_dir AutogluonModels/ag-20231026_024542  --pretrained_model_dir ./exps/pretrain_2023-10-30_22-10-52_din/output

# cr2: AutogluonModels/ag-20231030_092803   ./exps/pretrain_2023-10-30_16-12-24_din/output
python main_runner.py --task eval_prediction_after_pretraining --per_device_train_batch_size 250 --data_path ./data/cr2 --dataset_name cr2 --label_col "Binary Rating" --feature_transform_res_dir AutogluonModels/ag-20231030_092803  --pretrained_model_dir exps/pretrain_2023-10-30_16-12-24_din/output

#-------------
python main_runner.py --task pretrain --per_device_train_batch_size 250 --data_path ./data/cr --dataset_name cr --label_col "binaryRating" --feature_transform_res_dir AutogluonModels/ag-20231026_024542 --modality_fusion_method concat