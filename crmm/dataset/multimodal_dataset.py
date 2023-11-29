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
                 label_list=None,
                 class_weights=None):
        self.texts_list = texts_list
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [i for i in range(len(np.unique(labels)))]

    def __getitem__(self, idx):
        item = {
            'labels': torch.tensor(self.labels.iloc[idx]) if self.labels is not None else None,
            # note cat feature should be int tensor!
            'cat': self.cat_feats.iloc[idx] \
                if self.cat_feats is not None else torch.zeros(0),
            'num': torch.tensor(self.numerical_feats.iloc[idx]).float() \
                if self.numerical_feats is not None else torch.zeros(0),
            'text': self.texts_list.iloc[idx] if self.texts_list is not None else None
        }  # text_feats will be tokenized in MultimodalDatasetCollator
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        """returns the label names for classification"""
        return self.label_list


class MultimodalNormalCollator:

    def __init__(self, tokenizer, max_token_length) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __call__(self, features):
        texts = [f['text'].values[0] for f in features]
        text_encodings = self.tokenizer(texts, padding=True, truncation=True,
                                        max_length=self.max_token_length, return_tensors="pt")

        cats = ['[SEP]'.join(f['cat'].values) for f in features]
        cat_encodings = self.tokenizer(cats, padding=True, truncation=True,
                                       max_length=self.max_token_length, return_tensors="pt")

        nums = [f['num'] for f in features]

        labels = [f['labels'] for f in features]

        data = {
            'text': text_encodings.data,  # dict of input_ids and attention_mask
            'cat': cat_encodings.data,  # dict of input_ids and attention_mask
            'num': torch.stack(nums),
            'labels': torch.stack(labels)
        }

        return data


class MultimodalClipPairMatchCollator:

    def __init__(self, tokenizer, max_token_length=None, natural_language_labels=None) -> None:
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.natural_language_labels = natural_language_labels

    def __call__(self, features):
        text_encodings = self.tokenizer(self.natural_language_labels, padding=True, truncation=True,
                                        max_length=self.max_token_length, return_tensors="pt")

        cats = ['[SEP]'.join(f['cat'].values) for f in features]
        cat_encodings = self.tokenizer(cats, padding=True, truncation=True,
                                       max_length=self.max_token_length, return_tensors="pt")

        nums = [f['num'] for f in features]

        labels = [f['labels'] for f in features]

        data = {
            'text': text_encodings.data,  # dict of input_ids and attention_mask
            'cat': cat_encodings.data,  # dict of input_ids and attention_mask
            'num': torch.stack(nums),
            'labels': torch.stack(labels)
        }
        return data
