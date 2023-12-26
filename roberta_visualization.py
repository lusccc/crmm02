from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# 加载预训练的RoBERTa模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# 待分类的句子
sentences = ["This is an example sentence.", "Another example of a sentence."]

# 编码文本
encoded_dict = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# 将编码的文本输入模型
# 注意：没有标签提供给模型，因此这是一个推断步骤。
outputs = model(**encoded_dict)

# 从输出中获取logits
logits = outputs.logits

# 将logits转换为概率
probabilities = torch.nn.functional.softmax(logits, dim=1)

# 获取最可能的预测
predictions = torch.argmax(probabilities, dim=1)

# 输出预测结果
print("Predictions:", predictions)