# 统计数据集中样本text长度超过512的样本数量，以此来估计长文本占比，
# 假如截断它们，看对模型训练的影响大小
# 结论：超长样本占比3%，可以忽略
from ner_data import tokenizer
import json
max_seq_len = 512  # 定义阈值
count = 0  # 记录超过长度的样本数
total_samples = 1491  # 记录总样本数
long_text_samples = []  # 保存超长文本样本

data_file = "E:/NLP任务/关系抽取/drge/ori_data/train.json"
# 读取数据文件
with open(data_file, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line.strip())
        text = item['text']
        if item['spo_list']:  # 先排除掉空样本，在此基础上统计超长文本样本数
            # 分词
            tokenized = tokenizer(text, truncation=False, return_tensors='pt')
            # [0]:获取第一个分段，分段截取已经证明了超过512，取第一段只是为了做演示
            text_length = len(tokenized['input_ids'][0])  # 获取分词后的长度
            if text_length > 512:
                count += 1
                long_text_samples.append({
                    'text': text,
                    'length': text_length
                })

print(f"超长样本数量 (长度 > {512}): {count}")
print(f"超长样本占比: {count / total_samples:.2f}")

# 可选：打印超长样本的信息
for idx, sample in enumerate(long_text_samples[:]):
    print(f"样本 {idx + 1} 长度：{sample['length']}\n文本：{sample['text']}\n")