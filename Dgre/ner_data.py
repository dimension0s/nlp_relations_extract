# 批量处理数据集,基于原版，对ner数据集和re数据集分别分批处理
# 就算改完不如不改也没关系，是一坨屎也没关系，先练着写
import json
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader,random_split
from transformers import AutoTokenizer

class DgreNerData(Dataset):
    """提取ner数据"""
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = []
        with open(data_file,'r',encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                text = item['text']
                labels = []
                if item['spo_list']:
                    for spo in item['spo_list']:  # 过滤空样本，即spo_list:[]的状态，避免在后期的损失计算中出现错误
                        h,t = spo['h'],spo['t']
                        h_name,t_name = h['name'],t['name']
                        h_start,h_end = h['pos'][0],h['pos'][1]
                        t_start,t_end = t['pos'][0],t['pos'][1]
                        labels.append([h_start, h_end, h_name, '故障设备'])
                        labels.append([t_start, t_end, t_name, '故障原因'])
                    Data.append({
                        'text': text,
                        'labels': labels,
                    })
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

ner_data = DgreNerData("E:/NLP任务/关系抽取/drge/ori_data/train.json")
print(len(ner_data))  # 原数量是1491,过滤掉空样本后的总数量是1469
train_ner_data,dev_ner_data = random_split(ner_data,[1175,294])
print(train_ner_data[0])
print(dev_ner_data[0])

categories = {"故障设备","故障原因"}  # 使用集合字面量，去重并创建集合
id2label = {0:'O'}
for c in sorted(categories):
    id2label[len(id2label)] = f'B-{c}'
    id2label[len(id2label)] = f'I-{c}'

label2id = {v:k for k,v in id2label.items()}
print(id2label)  # {0: 'O', 1: 'B-故障原因', 2: 'I-故障原因', 3: 'B-故障设备', 4: 'I-故障设备'}
print(label2id)  # {'O': 0, 'B-故障原因': 1, 'I-故障原因': 2, 'B-故障设备': 3, 'I-故障设备': 4}

# 分批处理
checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
max_length = 512
def collate_fn(batch_samples):
    batch_sentence,batch_tags = [],[]
    for sample in batch_samples:
        batch_sentence.append(sample['text'])
        batch_tags.append(sample['labels'])
    batch_inputs = tokenizer(
        batch_sentence,
        max_length=max_length,  # 设置最大长度
        padding=True,  # 填充
        truncation=True,  # 截断
        return_tensors='pt',
    )
    batch_label = np.zeros(batch_inputs['input_ids'].shape,dtype=int)

    for s_idx,sentence in enumerate(batch_sentence):
        encoding = tokenizer(sentence,truncation=True,max_length=max_length)
        batch_label[s_idx][0] = -100  # 将[CLS]转成-100，不参与损失计算
        batch_label[s_idx][len(encoding.tokens())-1:] = -100  # 将[SEP]转成-100，不参与损失计算
        for char_start,char_end,_,tag in batch_tags[s_idx]:
            token_start = encoding.char_to_token(char_start)
            token_end = encoding.char_to_token(char_end)
            if token_start is None or token_end is None:
                # 目的是为了跳过超长文本，超长文本被截断后，部分标签会找不到对应的映射，在后面的损失计算中会产生错误
                continue
            batch_label[s_idx][token_start] = label2id[f'B-{tag}']
            batch_label[s_idx][token_start+1:token_end] = label2id[f'I-{tag}']
    return batch_inputs,torch.tensor(batch_label)

train_ner_dataloader = DataLoader(train_ner_data,batch_size=12,shuffle=True,collate_fn=collate_fn)
dev_ner_dataloader = DataLoader(dev_ner_data,batch_size=12,shuffle=False,collate_fn=collate_fn)

batch_X,batch_y = next(iter(train_ner_dataloader))
print('batch_X shape:',{k:v.shape for k,v in batch_X.items()})
print('batch_y shape:',batch_y.shape)
print(batch_X)
print(batch_y)


