# 数据预处理

# 1.提取events信息
def extract_events(events):
    """
    提取每个事件的触发词及事件类型。
    """
    event_dict = {}
    for event in events:
        event_id = event["id"]
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
            "event_type": event_type
        }
    return event_dict


# 2.提取relations信息
def extract_relations(relations, event_dict, text):
    """
    根据 relations 和 events 提取模型输入。
    """


    for relation in relations:
        # 获取事件对的触发词信息
        event1_id = relation["one_event_id"]
        event2_id = relation["other_event_id"]
        relation_type = relation["relation_type"]

        event1 = event_dict[event1_id]
        event2 = event_dict[event2_id]

        # 构造输入文本：包含触发词1、触发词2和原始文本
        input_text = f"[CLS] {event1['trigger_text']} [SEP] {event2['trigger_text']} [SEP] {text} [SEP]"


        # 将关系类型映射为标签
        if relation_type == "因果":
            label = 1
        elif relation_type == "时序":
            label = 0

    return input_text, label

# 整合到Dataset中
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split

class EventRelationDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["doc"]
        events = extract_events(item["events"])
        inputs, labels = extract_relations(item["relations"], events, text)

        return inputs, torch.tensor(labels)

# data = EventRelationDataset("E:/NLP项目/CCKS2024-面向篇章级文本的突发事件关系抽取评测/data.txt")
# print(data[0])
# print(len(data))

with open("E:/NLP项目/CCKS2024-面向篇章级文本的突发事件关系抽取评测/data.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for i, line in enumerate(lines):
        print(f"Line {i}: {line}")
