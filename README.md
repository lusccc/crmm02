## Description

This is the official implementation of the paper [LLM-infused bi-level semantic enhancement for corporate credit risk prediction](https://www.sciencedirect.com/science/article/pii/S0306457325000330).

### Abstract

Corporate credit risk (CCR) prediction enables investors, governments, and companies to make informed financial decisions. Existing research primarily focuses solely on the tabular feature values, yet it often overlooks the rich inherent semantic information. In this paper, a novel bi-level semantic enhancement framework for CCR prediction is proposed. Firstly, at the data-level, a large language model (LLM) generates detailed textual descriptions of companies' financial conditions, infusing raw tabular training data with semantic information and domain knowledge. Secondly, to enable semantic perception during inference when only tabular data is available, a contrastive multimodal multitask learning model (CMML) is proposed at the model level. CMML leverages the semantically enhanced data from the previous level to acquire semantic perception capabilities during the training phase, requiring only tabular data during prediction. It aligns the representations of tabular data with textual data, enabling extracting semantically rich features from tabular data. Furthermore, a semantic alignment classifier and an MLP classifier are integrated into a weighted ensemble learner within a multitask learning architecture to enhance robustness. Empirical verification on two datasets demonstrates that CMML surpasses benchmark models in key metrics, particularly in scenarios with limited samples and high proportions of unseen corporations, implying its effectiveness in CCR prediction through bi-level semantic enhancement.

## Directory Structure

```
.
├── crmm/                                # Core model package
│   ├── arguments.py                     # Argument definitions (CrmmTrainingArguments, LanguageModelArguments, MultimodalDataArguments)
│   ├── runner_setup.py                  # Experiment directory creation and logging config
│   ├── runner_callback.py               # Custom callbacks (contrastive early stopping, per-epoch evaluation)
│   ├── metrics.py                       # Evaluation metrics (acc, roc_auc, ks, gmean, type1/type2_acc, etc.)
│   ├── models/                          # Model implementations
│   │   ├── clncp.py                     # CLNCP model (Contrastive Language–Numeric-Category Pretraining)
│   │   ├── clncp_config.py             # CLNCP configuration class
│   │   ├── feature_extractor_factory.py # Feature extractor factory
│   │   ├── num_feature_extractor.py     # Numerical feature extractor
│   │   ├── cat_feature_extractor.py     # Categorical feature extractor
│   │   ├── text_feature_extractor.py    # Text feature extractor
│   │   ├── joint_feature_extractor.py   # Joint feature extractor (num+cat fusion)
│   │   ├── text_language_model.py       # Text language model (HuggingFace-based)
│   │   ├── num_cat_language_model.py    # Numeric-categorical language model
│   │   └── layer_utils.py              # Classifier and loss function utilities
│   ├── dataset/                         # Dataset processing
│   │   ├── data_info.py                # Dataset feature column definitions (cr/cr2)
│   │   ├── multimodal_data.py          # Multimodal data loading and preprocessing
│   │   ├── multimodal_dataset.py       # Dataset and Collator definitions
│   │   └── data_utils.py              # Data utilities
│   ├── gpt_description/                 # GPT semantic description generation
│   │   ├── semantic_query_poe.py       # Generate descriptions via Poe API
│   │   └── poe_prompt_example.txt      # GPT prompt example
│   └── utils/                           # General utilities
│       ├── utils.py                    # General utility functions
│       └── log_handler.py             # Log formatting
│
├── data/                                # Dataset directory (see data/README.md for details)
│   ├── cr/                              # CCR-S dataset (small-scale, ~1,830 records)
│   │   ├── all(with_description_col).csv  # Full dataset (with GPT descriptions)
│   │   ├── train/val/test.csv            # Random-split train/val/test sets
│   │   ├── train/val/test(with_description_col).csv  # Split sets with GPT descriptions
│   │   ├── train/val/test_description.csv             # GPT description files
│   │   ├── column_info.json             # Column information config
│   │   ├── cr-bpe.tokenizer.json        # BPE tokenizer
│   │   ├── cat_data.txt                 # Categorical column data (for tokenizer training)
│   │   ├── cat_tokenizer.py             # Tokenizer training script
│   │   ├── label_binaryzation.py        # Label binarization script
│   │   ├── merge_gpt_semantic_col.py    # Merge GPT description column
│   │   └── merge_to_all_with_description.py  # Generate full dataset with descriptions
│   ├── cr2/                             # CCR-L dataset (large-scale, ~7,806 records)
│   │   └── (same structure as cr/, plus tokenizer_test.py)
│   ├── dataset_split.py                 # Dataset random split script
│   └── README.md                        # Detailed dataset documentation
│
├── main_runner.py                       # Main training entry script
├── run_rolling.py                       # Rolling window experiments (Python version, recommended)
├── run_rolling.sh                       # Rolling window experiments (Shell version)
├── run_exps.py                          # Comprehensive experiment script (multiple experiment functions)
├── run_exp_demo.py                      # Demo run script
├── run_v1.sh                            # Early version experiment script
├── run_v2.sh                            # Version 2 experiment script
├── run_contrastive_early_stop.sh        # Contrastive early stopping experiments
├── run_unfreeze_part.sh                 # Partial unfreezing experiments
├── run_rolling_ml_benchmark.sh          # ML benchmark rolling window experiments
├── run_rolling_dl_benchmark.sh          # Deep learning benchmark rolling window experiments
│
├── ml_benchmark_model_comparison.py     # Traditional ML benchmark (LogR, SVM, KNN, DT, MLP, Adaboost, XGBoost, GBDT, RF)
├── pytorch_tabular_model_comparison.py  # Deep learning tabular benchmark (CategoryEmbedding, FTTransformer, TabNet, GATE, TabTransformer, AutoInt)
│
├── nlg_eval1.py                         # NLG evaluation: FactCC factual consistency detection
├── nlg_eval2.py                         # NLG evaluation script 2
├── nlg_eval4.py                         # NLG evaluation script 4
├── nlg_eva3.py                          # NLG evaluation script 3
├── BERTScore_results.xlsx               # BERTScore evaluation results
├── cr_factcc_results.xlsx               # CCR-S FactCC evaluation results
├── cr2_factcc_results.xlsx              # CCR-L FactCC evaluation results
│
├── excel_process/                       # Excel result post-processing
│   ├── get_final_metric_best_record.py  # Extract best metric records
│   ├── benchmark_avg.py                 # Benchmark result averaging
│   ├── fix_my_model_excel_result.py     # Fix Excel results
│   └── split_excel.py                   # Split Excel files
│
├── exp_visual/                          # Experiment visualization and statistical tests
│   ├── Friedman_Nemenyi_test.py         # Friedman-Nemenyi test
│   ├── nemenyi_plot_tool.py             # Nemenyi diagram plotting
│   ├── pair_t_test.py / pair_t_test_correct.py  # Paired t-test
│   ├── rolling_heatmap.py / rolling_heatmap_with_rank.py  # Rolling window heatmaps
│   ├── data_stat.py / data_stat_ks.py  # Data statistical analysis
│   ├── mds.py                           # MDS dimensionality reduction visualization
│   ├── dataset_info_visual.py           # Dataset info visualization
│   └── rolling_benchmark.py            # Rolling benchmark result processing
│
├── data_visual.ipynb                    # Data visualization notebook
├── metric_thresholds_test.ipynb         # Metric thresholds test notebook
├── roberta_visualization.py             # RoBERTa model visualization
├── cls_front_end_test.py               # Classification front-end test
│
├── exps/                                # Experiment output directory (auto-created)
├── excel/                               # Excel result output directory
├── hist_csv/                            # Per-epoch evaluation CSV output directory
├── saved_models/                        # Saved models directory
├── AutogluonModels/                     # AutoGluon models directory
│
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `pytorch_tabular_model_comparison.py` requires an additional `pytorch_tabular` dependency, which must be installed separately before running.

> **Important:** The project sets `TRANSFORMERS_OFFLINE=1` in `main_runner.py`, so language models must be downloaded to local cache beforehand. Before first run, either download models to the cache directory (default: `./exps/model_config_cache`), or change the environment variable to `0` to allow online downloading.

## Datasets

The project uses two corporate credit rating datasets. See `data/README.md` for details.

| Property | CCR-S (`data/cr/`) | CCR-L (`data/cr2/`) |
|------|------------|--------------|
| Total samples | ~1,830 | ~7,806 |
| Numerical features | 25 financial ratios | 16 financial indicators |
| Categorical features | 5 (Name, Symbol, Rating Agency Name, Sector, CIK) | 6 (Rating Agency, Corporation, CIK, SIC Code, Sector, Ticker) |
| Text feature | GPT_description | GPT_description |
| Label column | binaryRating (0/1) | Binary Rating (0/1) |
| Tokenizer | cr-bpe.tokenizer.json | cr2-bpe.tokenizer.json |

**Data split strategies:**
- **`random`**: First 80% for train+val, last 20% for test. Validation set is the first 10% of the train+val portion, then the remaining 90% becomes the actual training set. Final split: ~72% train, ~8% val, 20% test.
- **`rolling_window`**: Split by rating year via `--train_years` and `--test_years`, validation set is the first 10% of training-year data, the remaining 90% is the actual training set.

**GPT semantic descriptions:** The `GPT_description` column in the training set contains LLM-generated semantic descriptions of companies' financial conditions (generated via `crmm/gpt_description/semantic_query_poe.py`). Text columns in the test and validation sets are emptied to prevent data leakage — the model relies solely on tabular data during inference.

## Model Architecture

The core model is **CLNCP** (Contrastive Language–Numeric-Category Pretraining), inspired by the CLIP architecture, achieving contrastive alignment between tabular and text data.

**Three modality feature extractors:**
- **Numerical feature extractor** (`num`): Extracts numerical features via NumCatLanguageModel
- **Categorical feature extractor** (`cat`): Extracts categorical features via BPE tokenizer + NumCatLanguageModel
- **Text feature extractor** (`text`): Extracts text features via a pretrained language model (default: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`)

**Joint feature extractor** (`joint`): Concatenates `num` and `cat` embeddings, adds a learnable `[CLS]` token, then passes the combined sequence through NumCatLanguageModel (a BERT-like transformer). The `[CLS]` output serves as the fused tabular representation.

**Ensemble strategies** (`--clncp_ensemble_method`, **required** — the model will raise `ValueError` if not specified):
- `weighted_avg`: Learnable weighted average ensemble of two classifiers — the **NLL semantic alignment classifier** (contrastive similarity between tabular features and natural language label descriptions, e.g. "Poor credit"/"Good credit") and the **MLP classifier**. Total loss = contrastive_loss + cls_loss + ensemble_loss.
- `stacking`: Stacking ensemble of the same two classifiers via a 2-layer meta-learner. Total loss = contrastive_loss + cls_loss + ensemble_loss.
- `no_ensemble`: No ensemble, returns the MLP classifier logits directly. Total loss = contrastive_loss + cls_loss.
- `only_nll_pair_match`: Returns only the NLL contrastive logits (no MLP classifier output). Total loss = contrastive_loss only.

## Task Types

The project supports 6 task modes, specified via the `--task` parameter:

| Task | Model Mode | Description |
|--------|---------|------|
| `pretrain` | contrastive | Contrastive pretraining only (no labels needed), train+val sets are merged |
| `pair_match_evaluation` | contrastive_evaluation | Load pretrained model for contrastive evaluation (requires `--pretrained_model_dir`) |
| `finetune_classification` | classification | Load pretrained model for classification finetuning (requires `--pretrained_model_dir`) |
| `finetune_classification_scratch` | classification | Train classification from scratch (no pretraining) |
| `finetune_classification_evaluation` | classification | Load finetuned model for evaluation (requires `--finetuned_model_dir`) |
| `multi_task_classification` | contrastive_and_classification | **Recommended**: Joint contrastive + classification multitask training |

## Key Parameters

### Training Arguments (CrmmTrainingArguments)

| Parameter | Default | Description |
|------|--------|------|
| `--root_dir` | `./exps` | Experiment output root directory |
| `--task` | None | Task type (see table above) |
| `--data_path` | None | Dataset path, e.g. `./data/cr2` |
| `--dataset_name` | None | Dataset name: `cr` or `cr2` |
| `--dataset_split_strategy` | `random` | Data split strategy: `random` or `rolling_window` |
| `--train_years` | None | Training years (comma-separated), required for rolling_window |
| `--test_years` | None | Test years (comma-separated), required for rolling_window |
| `--use_modality` | None | Modalities to use, e.g. `num,cat,text` |
| `--fuse_modality` | None | Modalities to fuse, e.g. `num,cat` |
| `--contrastive_targets` | None | **Required for pretrain/multi_task.** Contrastive learning target pairs, e.g. `joint,text` or `num,text`. The first non-text item is aligned with `text` via CLIP-style contrastive loss |
| `--freeze_language_model_params` | True | Whether to freeze the text language model parameters |
| `--clncp_ensemble_method` | None | **Required.** Ensemble method: `weighted_avg`, `stacking`, `no_ensemble`, `only_nll_pair_match` |
| `--pretrained_model_dir` | None | Pretrained model directory (required for finetune/pair_match tasks) |
| `--finetuned_model_dir` | None | Finetuned model directory (required for evaluation tasks) |
| `--modality_fusion_method` | None | Modality fusion method (legacy parameter, stored in config but not actively used by current `JointFeatureExtractor`; kept for compatibility, e.g. `conv`) |
| `--contrastive_early_stopping_epoch` | None | Stop contrastive learning after this epoch |
| `--save_excel_path` | `./excel/res.xlsx` | Excel result save path |
| `--save_hist_eval_csv_path` | None | Per-epoch evaluation CSV path. When set: `evaluation_strategy` becomes `no`, `metric_for_best_model` becomes `loss`, and `EvaluateEveryNthEpoch` callback evaluates val+test every epoch |
| `--num_train_epochs` | 100 | Number of training epochs |
| `--per_device_train_batch_size` | 200 | Training batch size per device (auto-decreases on OOM since `auto_find_batch_size=True` by default) |
| `--patience` | 1000 | Early stopping patience (**currently inactive** — `EarlyStoppingCallback` is commented out in `main_runner.py`) |

### Language Model Arguments (LanguageModelArguments)

| Parameter | Default | Description |
|------|--------|------|
| `--language_model_name` | `''` | Pretrained language model name or path |
| `--max_seq_length` | 512 | Maximum sequence length |
| `--cache_dir` | `./exps/model_config_cache` | Model cache directory |
| `--load_hf_model_from_cache` | True | Whether to load model from local cache |
| `--load_hf_pretrained` | True | Whether to load pretrained weights |
| `--num_cat_language_model_hyperparameters` | `512,8,8,2048` | NumCatLanguageModel hyperparams (hidden_size,num_heads,num_layers,ffn_dim) |

### Data Arguments (MultimodalDataArguments)

| Parameter | Default | Description |
|------|--------|------|
| `--dataset_info` | `''` | Optional experiment description string, saved in Excel results for identification |
| `--num_train_samples` | None | Limit training samples (for few-shot experiments). For `random`: takes first N rows as train; for `rolling_window`: takes first N rows of training-year data as train, rest as val |
| `--use_val` | True | Whether to use a validation set |
| `--numerical_transformer_method` | `yeo_johnson` | Numerical preprocessing: `yeo_johnson`, `box_cox`, `quantile_normal`, `standard`, `none` |
| `--natural_language_labels` | `Poor credit@Good credit` | Natural language labels (separated by @, used for NLL semantic alignment classifier). E.g. `Poor credit@Good credit` → the text encoder encodes these as label prototypes for contrastive matching |

## Usage

### 1. Multi-task Classification (Recommended, Paper's Main Method)

Joint contrastive learning and classification multitask training:

```bash
python main_runner.py \
  --root_dir ./exps \
  --task "multi_task_classification" \
  --per_device_train_batch_size 300 \
  --num_train_epochs 50 \
  --data_path "./data/cr2" \
  --dataset_name "cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" \
  --test_years "2013" \
  --freeze_language_model_params True \
  --use_modality "num,cat,text" \
  --fuse_modality "num,cat" \
  --contrastive_targets "joint,text" \
  --clncp_ensemble_method "weighted_avg" \
  --num_cat_language_model_hyperparameters "512,8,8,2048" \
  --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" \
  --load_hf_model_from_cache True \
  --save_excel_path "./excel/cr2_res_rolling_window_#2010,2011,2012#_#2013#_multitask.xlsx"
```

### 2. Pretrain + Finetune (Two-stage Training)

Contrastive pretraining first, then classification finetuning:

```bash
# Stage 1: Pretraining
python main_runner.py \
  --root_dir ./exps \
  --task "pretrain" \
  --per_device_train_batch_size 128 \
  --num_train_epochs 20 \
  --data_path "./data/cr2" \
  --dataset_name "cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" \
  --test_years "2013" \
  --freeze_language_model_params True \
  --use_modality "num,cat,text" \
  --fuse_modality "num,cat" \
  --contrastive_targets "joint,text" \
  --clncp_ensemble_method "no_ensemble" \
  --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" \
  --load_hf_model_from_cache True

# Stage 2: Finetuning (--pretrained_model_dir points to Stage 1 output directory)
python main_runner.py \
  --root_dir ./exps \
  --task "finetune_classification" \
  --per_device_train_batch_size 128 \
  --num_train_epochs 100 \
  --data_path "./data/cr2" \
  --dataset_name "cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" \
  --test_years "2013" \
  --freeze_language_model_params False \
  --use_modality "num,cat,text" \
  --fuse_modality "num,cat" \
  --clncp_ensemble_method "no_ensemble" \
  --language_model_name "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis" \
  --load_hf_model_from_cache True \
  --pretrained_model_dir "exps/pretrain_<timestamp>_<rand>/output" \
  --save_excel_path "./excel/cr2_prefine_result.xlsx"
```

### 3. Scratch Classification (No Pretraining Baseline)

```bash
python main_runner.py \
  --root_dir ./exps \
  --task "finetune_classification_scratch" \
  --per_device_train_batch_size 1000 \
  --num_train_epochs 100 \
  --data_path "./data/cr2" \
  --dataset_name "cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" \
  --test_years "2013" \
  --use_modality "num,cat" \
  --fuse_modality "num,cat" \
  --clncp_ensemble_method "no_ensemble" \
  --save_excel_path "./excel/cr2_scratch_result.xlsx"
```

### 4. Benchmark Comparison

**ML benchmark models:**
```bash
python ml_benchmark_model_comparison.py \
  --dataset_name cr2 --data_path "./data/cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" --test_years "2013" \
  --excel_path "./excel/cr2_ml_benchmark.xlsx"
```

**Deep learning tabular benchmark models:**
```bash
python pytorch_tabular_model_comparison.py \
  --dataset_name cr2 --data_path "./data/cr2" \
  --dataset_split_strategy "rolling_window" \
  --train_years "2010,2011,2012" --test_years "2013" \
  --excel_path "./excel/cr2_dl_benchmark.xlsx"
```

### 5. Batch Experiment Scripts

The project provides multiple batch experiment scripts:

| Script | Description |
|------|------|
| `run_rolling.py` / `run_rolling.sh` | **Recommended**: Rolling window batch experiments with multi-config support |
| `run_exps.py` | Comprehensive experiment script with `run_pre_fine_rolling_window`, `run_multitask_rolling_window`, `run_benchmark_*` functions |
| `run_exp_demo.py` | Simple demo script using `run_exps.CommandRunner` |
| `run_v2.sh` | Various epoch/config experiment commands |
| `run_v1.sh` | Early version experiment commands (contains legacy task names) |
| `run_contrastive_early_stop.sh` | Contrastive early stopping experiments (sweep over different stopping epochs) |
| `run_unfreeze_part.sh` | Partial language model unfreezing experiments |
| `run_rolling_ml_benchmark.sh` | ML benchmark rolling window batch experiments |
| `run_rolling_dl_benchmark.sh` | Deep learning benchmark rolling window batch experiments |

### 6. NLG Evaluation

Evaluate the quality of GPT-generated descriptions:

```bash
python nlg_eval1.py   # FactCC factual consistency detection
python nlg_eval2.py   # Other NLG evaluation metrics
```

## Output

- **`exps/`**: Each experiment auto-creates a subdirectory `exps/<task>_<timestamp>_<rand>/`, containing:
  - `output/`: Model checkpoints, training argument JSONs, log files
  - `logging/`: TensorBoard logs
- **`excel/`**: Evaluation result Excel files (with val/test acc, roc_auc, ks, gmean, type1_acc, type2_acc, etc.)
- **`hist_csv/`**: Per-epoch evaluation metric CSV files (requires `--save_hist_eval_csv_path`)

## Evaluation Metrics

- **acc**: Accuracy
- **roc_auc**: ROC AUC
- **ks**: KS statistic (max of TPR - FPR)
- **gmean**: Geometric mean (√(type1_acc × type2_acc))
- **type1_acc**: Specificity (TN/(TN+FP))
- **type2_acc**: Recall/Sensitivity (TP/(TP+FN))
- **f1**: F1 score
- **pr_auc**: PR curve area

### How to Cite
If you find this work useful in your research, please consider citing:
```bibtex
@article{LU2025104091,
    title = {LLM-infused bi-level semantic enhancement for corporate credit risk prediction},
    journal = {Information Processing & Management},
    volume = {62},
    number = {4},
    pages = {104091},
    year = {2025},
    issn = {0306-4573},
    doi = {https://doi.org/10.1016/j.ipm.2025.104091},
    url = {https://www.sciencedirect.com/science/article/pii/S0306457325000330},
    author = {Sichong Lu and Yi Su and Xiaoming Zhang and Jiahui Chai and Lean Yu},
}