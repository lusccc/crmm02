import torch
from transformers import BertModel, PreTrainedTokenizerFast

# Load your tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file="cr2-bpe.tokenizer.json")


# 训练完成后，获取vocab_size
vocab_size = tokenizer.vocab_size

print(f"Vocab size: {vocab_size}")


# 获取词汇表中的所有tokens
vocab = tokenizer.get_vocab()

# 打印词汇表中的tokens和它们对应的索引
for token, index in vocab.items():
    print(f"{token}: {index}")



print()

# Load the pretrained BERT model
# model = BertModel.from_pretrained('bert-base-uncased')

# Get the IDs for [CLS] and [SEP] tokens from the tokenizer
cls_token_id = tokenizer.cls_token_id  # should be 1 from your tokenizer config
sep_token_id = tokenizer.sep_token_id  # should be 2 from your tokenizer config

# Assuming num_embeds and cat_embeds are tensors containing your embeddings
# num_embeds = ...
# cat_embeds = ...

# Get the embeddings for [CLS] and [SEP] from the BERT embedding layer
cls_embeds = model.embeddings.word_embeddings(torch.tensor([cls_token_id]))
sep_embeds = model.embeddings.word_embeddings(torch.tensor([sep_token_id]))

# Make sure the special token embeddings are expanded to match your batch size
cls_embeds = cls_embeds.expand(num_embeds.size(0), -1, -1)
sep_embeds = sep_embeds.expand(cat_embeds.size(0), -1, -1)

# Concatenate the embeddings
embeds = torch.cat([cls_embeds, num_embeds, sep_embeds, cat_embeds], dim=1)

# Pass the embeddings to the BERT model
output = model(inputs_embeds=embeds)