import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

cat_cols = ['Rating Agency', 'Corporation', 'CIK', 'SIC Code', 'Sector', 'Ticker']
# 读取CSV文件
data = pd.read_csv('train.csv')
for col in ['CIK', 'SIC Code']:
    if col in data.columns:
        data[col] = data[col].fillna(-1).astype(int).astype(str)

# 提取指定列的数据
selected_data = data[cat_cols]

# 保存为TXT文件
selected_data.to_csv('cat_data.txt', sep='\t', index=False)

# 初始化一个基于Byte Pair Encoding (BPE) 的tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 初始化一个trainer，用于训练BPE模型
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# 初始化一个pre-tokenizer，用于数据的初步分割
tokenizer.pre_tokenizer = Whitespace()

# 准备您的数据路径：可以是一个列表，包含多个文件
files = ["cat_data.txt"]

# 开始训练
tokenizer.train(files, trainer)

# 保存训练好的tokenizer到磁盘
tokenizer.save("cr2-bpe.tokenizer.json")