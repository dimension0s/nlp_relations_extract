# 模型训练，验证，主循环
import os
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import numpy as np
from model import ner_model
from ner_data import train_ner_data,dev_ner_data,train_ner_dataloader,dev_ner_dataloader
# seqeval 是一个专门用于序列标注评估的 Python 库，支持 IOB、IOB、IOBES 等多种标注格式以及多种评估策略
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from device import device
from ner_data import id2label

# 1.训练函数
def train_loop(dataloader,model,loss_fn,optimizer,lr_scheduler,epoch):
    total_loss = 0.
    correct = 0.
    total = 0
    model.train()

    progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step,(X,y) in progress_bar:
        X, y = X.to(device),y.to(device)
        pred = model(X)
        loss = loss_fn(pred.permute(0,2,1),y.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss/(step+1)

        progress_bar.set_description(f'epoch {epoch},loss:{avg_loss:.>7f}')
    return total_loss


# 2.验证/测试函数
def test_loop(dataloader,model,mode='Test'):
    assert mode in ['Valid','Test']
    true_labels,true_predictions = [],[]
    model.eval()
    with torch.no_grad():
        for X,y in tqdm(dataloader):
            pred = model(X.to(device))
            # 找出预测值中的最优对应的索引
            predictions = pred.argmax(dim=-1).cpu().numpy().tolist()
            labels = y.cpu().numpy().tolist()  # 真实值
            true_labels += [[id2label[int(l)] for l in label if l != -100]
                            for label in labels]
            true_predictions += [
                [id2label[int(p)] for (p,l) in zip(prediction,label) if l != -100]
                for prediction,label in zip(predictions,labels)
            ]
    print(classification_report(true_labels,true_predictions,mode='strict',scheme=IOB2))
    return classification_report(
        true_labels,
        true_predictions,
        mode='strict',
        output_dict=True,
    )

# 3.主循环
from transformers import AdamW,get_scheduler
import random

learning_rate = 2e-5  # 原来是1e-5，不行再调
batch_size = 4
epoch_num = 7

def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = AdamW(ner_model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_ner_dataloader)
)

total_loss = 0.
best_f1 = 0.
for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_ner_dataloader, ner_model, loss_fn, optimizer, lr_scheduler, epoch + 1)
    metrics = test_loop(dev_ner_dataloader, ner_model, 'Valid')
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        print('saving new weights...\n')
        torch.save(
            ner_model.state_dict(),
            f'epoch_{epoch + 1}_valid_macro_f1_{(100 * valid_macro_f1):0.3f}_micro_f1_'
            f'{(100 * valid_micro_f1):0.3f}_weights.bin')



