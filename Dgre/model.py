# 模型构建：基于BERT构建自定义模型

import torch
import torch.nn as nn
from transformers import BertModel,BertConfig,BertPreTrainedModel,AutoConfig
from ner_data import id2label,checkpoint
from device import device

# 1.NER模型
class BertNER(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config,add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 一般是0.1
        self.cls = nn.Linear(768,5)  # [768,5]
        self.post_init()  # 后处理，即参数初始化

    def forward(self,x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.last_hidden_state  # 或者bert_output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
ner_model = BertNER.from_pretrained(checkpoint,config=config).to(device)
print("ner model:\n",ner_model)

# 2.RE模型
from re_data import label2id
class BertRE(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.bert = BertModel(config,add_pooling_layer=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 一般是0.1
        self.cls = nn.Linear(768,len(label2id))  # [768,]
        self.post_init()  # 后处理，即参数初始化

    def forward(self,x):
        bert_output = self.bert(**x)
        sequence_output = bert_output.pooler_output  # 等价于bert_output[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.cls(sequence_output)
        return logits

config = AutoConfig.from_pretrained(checkpoint)
re_model = BertRE.from_pretrained(checkpoint,config=config).to(device)
print("re model:\n",re_model)

