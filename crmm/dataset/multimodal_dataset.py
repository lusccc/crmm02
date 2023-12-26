import numpy as np
import torch
from torch.utils.data import Dataset


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


class MultimodalCollator:

    def __init__(self, text_tokenizer, cat_tokenizer, max_token_length, natural_language_labels) -> None:
        # text_tokenizer.pad_token = tokenizer.eos_token   #  for GPT2! add this line
        self.text_tokenizer = text_tokenizer
        self.max_token_length = max_token_length
        self.cat_tokenizer = cat_tokenizer  # also used in joint feature extractor
        self.natural_language_labels = natural_language_labels

    def __call__(self, features):
        text_encodings = None
        nll_encodings = None
        if self.text_tokenizer is not None:
            texts = [f['text'].values[0] for f in features]
            text_encodings = self.text_tokenizer(texts, padding=True, truncation=True,
                                                 max_length=self.max_token_length, return_tensors="pt")
            nll_encodings = self.text_tokenizer(self.natural_language_labels, padding=True, truncation=True,
                                                max_length=self.max_token_length, return_tensors="pt")

        cat_encodings = None
        if self.cat_tokenizer is not None:
            # @@@ try 1: direct concat
            cats = ['[SEP]'.join(f['cat'].values) for f in features]

            # @@@ try 2: make sentence
            # for cr:
            # template = "The company %s with symbol %s and CIK %s is rated by %s in the sector of %s."
            # for cr2:
            # template = ("%s, assigned CIK %s and operating in the %s sector with SIC Code %s, "
            #             "is rated by %s and trades with the ticker symbol %s.")
            # cats = []
            # for f in features:
            #     cat_f = f['cat']
            #     # for cr:
            #     # cat_sentence = template % (cat_f['Name'],
            #     #                            cat_f['Symbol'],
            #     #                            cat_f['CIK'],
            #     #                            cat_f['Rating Agency Name'],
            #     #                            cat_f['Sector'])
            #     # for cr2:
            #     cat_sentence = template % (cat_f['Corporation'],
            #                                cat_f['CIK'],
            #                                cat_f['Sector'],
            #                                cat_f['SIC Code'],
            #                                cat_f['Rating Agency'],
            #                                cat_f['Ticker'])
            #     cats.append(cat_sentence)

            cat_encodings = self.cat_tokenizer(cats, padding=True, truncation=True,
                                               max_length=self.max_token_length, return_tensors="pt")

        nums = [f['num'] for f in features]

        labels = [f['labels'] for f in features]

        data = {
            'text': text_encodings.data if text_encodings is not None else None,
            'nll': nll_encodings.data if nll_encodings is not None else None,
            'cat': cat_encodings.data if cat_encodings is not None else None,
            'num': torch.stack(nums),
            'labels': torch.stack(labels)
        }
        return data

