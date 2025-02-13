## Description

This is the official implementation of the paper [LLM-infused bi-level semantic enhancement for corporate credit risk prediction](https://www.sciencedirect.com/science/article/pii/S0306457325000330).

### Abstract

Corporate credit risk (CCR) prediction enables investors, governments, and companies to make informed financial decisions. Existing research primarily focuses solely on the tabular feature values, yet it often overlooks the rich inherent semantic information. In this paper, a novel bi-level semantic enhancement framework for CCR prediction is proposed. Firstly, at the data-level, a large language model (LLM) generates detailed textual descriptions of companies’ financial conditions, infusing raw tabular training data with semantic information and domain knowledge. Secondly, to enable semantic perception during inference when only tabular data is available, a contrastive multimodal multitask learning model (CMML) is proposed at the model level. CMML leverages the semantically enhanced data from the previous level to acquire semantic perception capabilities during the training phase, requiring only tabular data during prediction. It aligns the representations of tabular data with textual data, enabling extracting semantically rich features from tabular data. Furthermore, a semantic alignment classifier and an MLP classifier are integrated into a weighted ensemble learner within a multitask learning architecture to enhance robustness. Empirical verification on two datasets demonstrates that CMML surpasses benchmark models in key metrics, particularly in scenarios with limited samples and high proportions of unseen corporations, implying its effectiveness in CCR prediction through bi-level semantic enhancement.

## Directory Structure

```
.
├── crmm/                   # Core model implementation
├── data/                   # Dataset directories
│   ├── cr/                # Credit Rating dataset 1
│   └── cr2/               # Credit Rating dataset 2
├── excel_process/         # Excel result processing
├── exp_visual/           # Experiment visualization
├── main_runner.py        # Main training script
├── benchmark_model_comparison.py  # ML model benchmarking
└── requirements.txt      # Dependencies
```


## Installation

```bash
pip install -r requirements.txt
```

### Model Training

1. Pretraining stage:
```bash
python main_runner.py --task pretrain \
    --data_path ./data/cr2 \
    --dataset_name cr2 \
    --dataset_split_strategy rolling_window \
    --train_years "2010,2011,2012" \
    --test_years "2013" \
    --use_modality "num,cat,text"
```


2. Finetuning stage:
```bash
python main_runner.py --task finetune_classification \
    --data_path ./data/cr2 \
    --pretrained_model_dir "path/to/pretrained/model" \
    --dataset_split_strategy rolling_window
```

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
````