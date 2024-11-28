# 批量处理数据集,基于原版，对ner数据集和re数据集分别分批处理
# 就算改完不如不改也没关系，是一坨屎也没关系，先练着写
import json
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader,random_split
from transformers import AutoTokenizer

class DgreReData(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = []
        with open(data_file,'r',encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                text = item['text']
                labels = item['labels']
                Data.append({
                    'text': text,
                    'labels': labels,
                })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_re_data = DgreReData("E:/NLP任务/关系抽取/drge/re_data/train.txt")
dev_re_data = DgreReData("E:/NLP任务/关系抽取/drge/re_data/dev.txt")
print(len(train_re_data))  # 8027
print(len(dev_re_data))   # 699
print(train_re_data[0])
print(dev_re_data[0])


# 构建标签映射字典
label_file = "E:/NLP任务/关系抽取/drge/re_data/labels.txt"
with open(label_file,'r',encoding='utf-8') as f:
    labels = f.read().strip().split('\n')
labels_num = len(labels)
label2id = {label:i for i,label in enumerate(labels)}
id2label = {i:label for i,label in enumerate(labels)}
print(label2id)  # {'检测工具': 0, '性能故障': 1, '部件故障': 2, '组成': 3, '没关系': 4}
print(id2label)  # {0: '检测工具', 1: '性能故障', 2: '部件故障', 3: '组成', 4: '没关系'}

# 分批处理
checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
max_length = 512
def collate_fn(batch_samples):
    batch_sentence,batch_labels = [],[]
    for sample in batch_samples:

        labels = sample['labels']
        if len(labels) != 3:
            continue
        h, t, label = labels
        text = sample['text']
        if h not in text or t not in text:
            continue
        label = label2id[label]
        batch_labels.append(label)
        # 在text前加上h,t，增强模型训练的准确性，本质是为分类模型提供上下文信息
        batch_sentence.append('[CLS]'+h+'[SEP]'+t+"[SEP]" + text + "[SEP]")

    batch_inputs = tokenizer(
        batch_sentence,
        max_length=max_length,  # 设置最大长度
        padding='max_length',  # 填充
        truncation=True,  # 截断
        return_tensors='pt',

    )
    return batch_inputs, torch.tensor(batch_labels, dtype=torch.long)

train_re_dataloader = DataLoader(train_re_data,batch_size=12,shuffle=True,collate_fn=collate_fn)
dev_re_dataloader = DataLoader(dev_re_data,batch_size=12,shuffle=False,collate_fn=collate_fn)

batch_X,batch_y = next(iter(train_re_dataloader))
print('batch_X shape:',{k:v.shape for k,v in batch_X.items()})
print('batch_y shape:',batch_y.shape)
print(batch_X)
print(batch_y)


