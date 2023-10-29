import numpy as np
import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import torch_default_data_collator


class MultimodalDataset(Dataset):

    def __init__(self,
                 texts_list,
                 categorical_feats,
                 numerical_feats,
                 labels=None,
                 df=None,
                 label_list=None,
                 class_weights=None):
        self.df = df
        self.texts_list = texts_list
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [i for i in range(len(np.unique(labels)))]

    def __getitem__(self, idx):
        item = {
            'labels': torch.tensor(self.labels[idx]) if self.labels is not None else None,
                # note cat feature should be int tensor!
                'cat': torch.tensor(self.cat_feats[idx]).int() \
                    if self.cat_feats is not None else torch.zeros(0),
                'num': torch.tensor(self.numerical_feats[idx]).float() \
                    if self.numerical_feats is not None else torch.zeros(0),
                'text': self.texts_list[idx] if self.texts_list else None
        }  # text_feats will be tokenized in MultimodalDatasetCollator
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification"""
        return self.label_list


class MultimodalPretrainCollator:

    def __init__(self, tokenizer, max_token_length=None) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __call__(self, features):
        texts = [f['text'] for f in features]
        tokenized = self.tokenizer(texts, padding=True, truncation=True,
                                   max_length=self.max_token_length, return_tensors="pt")
        # reorganize to dict
        t_keys = list(tokenized.keys())
        if 'token_type_ids' in t_keys:
            t_keys.remove('token_type_ids')  # ignore token_type_ids, since we don't use sequence pair
        for i, f in enumerate(features):
            for k in t_keys:
                f[k] = tokenized[k][i]

        features = torch_default_data_collator(features)
        features['text'] = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        del features['input_ids']
        del features['attention_mask']
        return features


class MultimodalClassificationCollator:

    def __init__(self, tokenizer, max_token_length=None, natural_language_labels=None) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.natural_language_labels = natural_language_labels

    def __call__(self, features):
        features = torch_default_data_collator(features)
        tokenized = self.tokenizer(self.natural_language_labels, padding=True, truncation=True,
                                   max_length=self.max_token_length, return_tensors="pt")
        features['text'] = {'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask']}
        return features
