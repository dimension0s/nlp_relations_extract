# 1.数据集预处理
# 1.1）加载数据集
# 输入文本格式：触发词1+触发词2+原始文本，标签：关系类型：时序，因果

import json

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SuddenEventData(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = []
        inputs,labels =[],[]
        # 逐行读取并按三行一组解析
        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 生成一个从 0 到 len(lines) 的步长为 3 的整数序列,
            # 因为对于lines,每三行为一条样本数据：text,events,relations
            for i in range(0,len(lines),2):
                # 分别解析文本行、事件行和关系行
                item = json.loads(lines[i].strip())
                relations_line = json.loads(lines[i+1].strip())
                new_text = item['doc']
                events = item['events']
                # 构造事件字典
                event_dict = {}
                for event in events:
                    event_id = event['id']
                    trigger = event["event-information"]["trigger"][0]  # 假设每个事件只有一个触发词
                    trigger_text = trigger["text"]
                    trigger_start = trigger["start"]
                    trigger_end = trigger["end"]
                    event_type = event["event-information"]["event_type"]

                    # 以事件ID为键，存储事件的触发词和类型
                    event_dict[event_id] = {
                        "trigger_text": trigger_text,
                        "trigger_start": trigger_start,
                        "trigger_end": trigger_end,
                        "event_type": event_type,
                    }
                # 解析关系数据
                relations = relations_line['relations']
                for relation in relations:
                    # 获取事件对的触发词信息
                    event1_id = relation["one_event_id"]
                    event2_id = relation["other_event_id"]
                    rel_type = relation["relation_type"]

                    event1 = event_dict[event1_id]
                    event2 = event_dict[event2_id]

                    # 构造输入文本：包含触发词1、触发词2和原始文本
                    input_text = f"{event1['trigger_text']} [SEP] {event2['trigger_text']} [SEP] {new_text}"
                    # inputs.append(input_text)

                    # 将关系类型映射为标签
                    if rel_type == "因果":
                        label = 1
                    elif rel_type == "时序":
                        label = 0

                    Data.append({
                        'text': input_text,
                        'label': label,
                    })
        # 返回输入和标签对
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 在这里，一条数据集的text由于event_type的不同被分成了若干条样本
data = SuddenEventData("E:/NLP项目/CCKS2024-面向篇章级文本的突发事件关系抽取评测/data.txt")
print(len(data))
print(data[1])

from transformers import AutoTokenizer

checkpoint = 'bert-base-chinese'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def collate(batch_samples):
    batch_text,batch_labels = [],[]
    for sample in batch_samples:
        batch_text.append(sample['text'])
        batch_labels.append(sample['label'])
    batch_inputs = tokenizer(
        batch_text,
        max_length=300,
        padding='max_length',
        truncation=True,
        stride=50,  # 超长文本分块处理
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    batch_labels = torch.tensor(batch_labels)

    return batch_inputs,batch_labels

train_dataloader = DataLoader(data,batch_size=4,shuffle=True,collate_fn=collate)
batch_X,batch_y = next(iter(train_dataloader))
print('batch_X shape:',{k:v.shape for k,v in batch_X.items()})
print('batch_y shape:',batch_y.shape)
print(batch_X)
print(batch_y)

