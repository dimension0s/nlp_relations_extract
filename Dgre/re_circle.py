# 模型训练，验证，主循环
import os
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import numpy as np
from model import re_model
from re_data import train_re_data,train_re_dataloader,dev_re_data,dev_re_dataloader
from sklearn.metrics import classification_report
from device import device
from re_data import id2label,label2id

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
        loss = loss_fn(pred,y)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            true_predictions.extend(predictions)
            true_labels.extend(labels)
    target_names = list(id2label.values())
    report = classification_report(true_predictions,true_labels,target_names=target_names)
    print(report)
    return report


# 3.主循环
from transformers import AdamW,get_scheduler
import random

learning_rate = 3e-5  # 原来是1e-5，不行再调
epoch_num = 5

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
optimizer = AdamW(re_model.parameters(),lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    # num_warmup_steps=int(0.05 * epoch_num * len(train_re_dataloader)),
    num_training_steps=epoch_num*len(train_re_dataloader)
)


best_f1 = 0.
best_model_path = None
for epoch in range(epoch_num):
    print(f"Epoch {epoch+1}/{epoch_num}\n-------------------------------")
    total_loss = train_loop(train_re_dataloader, re_model, loss_fn, optimizer, lr_scheduler, epoch + 1)
    metrics = test_loop(dev_re_dataloader, re_model, 'Valid')
    valid_macro_f1, valid_micro_f1 = metrics['macro avg']['f1-score'], metrics['micro avg']['f1-score']
    valid_f1 = metrics['weighted avg']['f1-score']
    if valid_f1 > best_f1:
        best_f1 = valid_f1
        if best_model_path:
            os.remove(best_model_path)  # 删除旧模型
        best_model_path = f'model_epoch_{epoch + 1}_f1_{valid_f1:.4f}.bin'
        print(f"Saving new best model to {best_model_path}\n")
        torch.save(
            re_model.state_dict(),best_model_path)

