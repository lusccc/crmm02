

for epoch in  10 20 30 40 50 60 70 80 90 100; do
  python main_runner.py --root_dir ./exps --task "multi_task_classification" --per_device_train_batch_size 160 --num_train_epochs $epoch --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2010,2011,2012" --test_years "2013" --freeze_language_model_params True --use_modality "num,cat,text" --fuse_modality "num,cat" --contrastive_targets "joint,text" --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_epoch_test.xlsx" --weight_decay 0.0001 --lr_scheduler_type "linear"
done


python main_runner.py --root_dir ./exps --task "multi_task_classification" --per_device_train_batch_size 176 --num_train_epochs 100 --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2010,2011,2012" --test_years "2013" --freeze_language_model_params True --use_modality "num,cat,text" --fuse_modality "num,cat" --contrastive_targets "joint,text" --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_epoch_test.xlsx" --save_hist_eval_csv_path "./hist_csv/hist_eval_res_cr2_2013_noclspademb_stacking_numLN.csv"

python main_runner.py --root_dir ./exps --task "multi_task_classification" --per_device_train_batch_size 144 --num_train_epochs 200 --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2011,2012,2013" --test_years "2014" --freeze_language_model_params True --use_modality "num,cat,text" --fuse_modality "num,cat" --contrastive_targets "joint,text" --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --save_excel_path "./excel/cr2_res_rolling_window_#2011,2012,2013#_#2014#_1225.xlsx" --save_hist_eval_csv_path "./hist_csv/hist_eval_res_cr2_2014_noclspademb_stacking_numLN_addclstoken.csv"


python main_runner.py --root_dir ./exps --task "finetune_classification_scratch" --per_device_train_batch_size 2000 --num_train_epochs 200 --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2011,2012,2013" --test_years "2014" --use_modality "num,cat" --fuse_modality "num,cat"  --save_excel_path "./excel/cr2_res_rolling_window_#2011,2012,2013#_#2014#_epoch_test.xlsx" --save_hist_eval_csv_path "./hist_csv/hist_eval_res_cr2_2014_scratch_numLN_addclstoken_hidd768ff3072_.csv"







python main_runner.py --root_dir ./exps --task "finetune_classification_scratch" --per_device_train_batch_size 1536 --num_train_epochs 200 --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2010,2011,2012" --test_years "2013" --use_modality "num,cat" --fuse_modality "num,cat"  --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_epoch_test.xlsx" --save_hist_eval_csv_path "./hist_csv/hist_eval_res_cr2_2013_scratch_numLayernormIndpFeatLayerEmb_jointDrp.5AddClsToken_bertcnf512-6-8-2048-vocabsize_fixWrongPadId.csv"


python main_runner.py --root_dir ./exps --task "multi_task_classification" --per_device_train_batch_size 144 --num_train_epochs 100 --data_path "./data/cr2" --dataset_name "cr2" --dataset_split_strategy "rolling_window" --train_years "2010,2011,2012" --test_years "2013" --freeze_language_model_params True --use_modality "num,cat,text" --fuse_modality "num,cat" --contrastive_targets "joint,text" --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" --load_hf_model_from_cache True --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_1226.xlsx" --save_hist_eval_csv_path "./hist_csv/hist_eval_res_cr2_2013_multitask_numLayernormIndpFeatLayerEmb_jointDrp.5AddClsToken_clncpThreeLoss_bertcnf512-6-8-2048-vocabsize_fixWrongPadId.csv"






























python main_runner.py --root_dir ./exps --task pretrain --per_device_train_batch_size 128 --num_train_epochs 20 --data_path ./data/cr2 --dataset_name cr2 --dataset_split_strategy rolling_window --modality_fusion_method conv --freeze_language_model_params True --use_modality num,cat,text --fuse_modality num,cat --language_model_name mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis --load_hf_model_from_cache True --save_excel_path ./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_prefine_numfreezeinfine_epoch_100_rep_1_freezeall_roberta_catemb_numtextcontrastive_promptconcat_tinystruct_onvalacc.xlsx --train_years 2010,2011,2012 --test_years 2013 --contrastive_targets "joint,text"

#--pretrained_model_dir "exps/pretrain_2023-12-20_20-37-06_faN/output"
#exps/pretrain_2023-12-20_20-37-06_faN/output

python main_runner.py --root_dir ./exps --task pair_match_evaluation --per_device_train_batch_size 128 --num_train_epochs 1 --data_path ./data/cr2 --dataset_name cr2 --dataset_split_strategy rolling_window --train_years 2010,2011,2012 --test_years 2013 --language_model_name mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis --load_hf_model_from_cache True --modality_fusion_method "conv" --use_modality "num,cat" --fuse_modality "num,cat" --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_pre_pairmatch_epoch_100_rep_1_freezeall_roberta_catemb_numtextcontrastive_promptconcat_tinystruct_onvalacc.xlsx" --pretrained_model_dir "exps/pretrain_2023-12-21_09-38-43_ygl/output"  --natural_language_labels "Poor credit@Good credit" --contrastive_targets "joint,text"


python main_runner.py --root_dir ./exps --task finetune_classification --per_device_train_batch_size 128 --num_train_epochs 1 --data_path ./data/cr2 --dataset_name cr2 --language_model_name mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis --load_hf_model_from_cache True --dataset_split_strategy rolling_window --train_years 2010,2011,2012 --test_years 2013 --modality_fusion_method "conv" --use_modality "num,cat" --fuse_modality "num,cat" --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_prefine_numfreezeinfine_epoch_100_rep_1_freezeall_roberta_catemb_numtextcontrastive_promptconcat_tinystruct_onvalacc.xlsx" --pretrained_model_dir "exps/pretrain_2023-12-21_09-38-43_ygl/output"


