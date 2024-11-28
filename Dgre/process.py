# 在关系抽取任务中，通常会分成NER命名实体识别抽取和关系抽取，即将抽取到的实体与关系组合成三元组
import os
import re
import json
import codecs
import random
from tqdm.auto import tqdm
from collections import defaultdict

# 数据预处理：分别处理成ner数据集和re（关系）数据集
# 1.Dgre的数据集处理
class ProcessDgreData:
    def __init__(self):
        self.data_path = 'E:/NLP任务/关系抽取/drge/'  # 初始数据文件夹
        self.train_file = self.data_path+'ori_data/train.json'  # 文件夹中的具体数据

    def get_ner_data(self):
        """从原数据集中提取NER数据，用于命名实体识别"""
        # 读取全部数据
        with open(self.train_file,'r',encoding='utf-8',errors='replace') as f:
            data = f.readlines()
        res = []
        for did,d in enumerate(data):
            d = eval(d)
            tmp = {}
            text = d['text']
            tmp['id'] = d['ID']
            tmp['text'] = [i for i in text]  # 将text按字符分割
            tmp['labels'] = ['O']*len(tmp['text'])  # 标签与文本text长度对齐
            for rel_id, spo in enumerate(d['spo_list']):
                h = spo['h']
                t = spo['t']
                h_start, h_end = h['pos'][0], h['pos'][1]
                t_start, t_end = t['pos'][0], t['pos'][1]
                tmp['labels'][h_start] = 'B-故障设备'
                for i in range(h_start+1,h_end):
                    tmp['labels'][i] = 'I-故障设备'
                tmp['labels'][t_start] = 'B-故障原因'
                for i in range(t_start+1,t_end):
                    tmp['labels'][i] = 'I-故障原因'
            res.append(tmp)  # 将处理好的数据加到res中（最终的呈现效果）
        # 分割数据集,因为原始数据集只有train集，所以要将其分隔为训练集和验证集
        train_ratio = 0.92
        train_num = int(len(res)*train_ratio)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        # 将处理好的数据存放到新的文件中:train.txt,dev.txt,labels.txt
        with open(self.data_path+'ner_data/train.txt','w',encoding='utf-8') as f:
            f.write('\n'.join([json.dumps(d,ensure_ascii=False) for d in train_data]))

        with open(self.data_path+'ner_data/dev.txt','w',encoding='utf-8') as f:
            f.write('\n'.join([json.dumps(d,ensure_ascii=False) for d in dev_data]))

        labels = ['故障设备','故障原因']
        with open(self.data_path+'ner_data/labels.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(labels))

    def get_re_data(self):
        """从原数据集中提取关系数据：RE，用于关系抽取"""
        # 读取原始数据
        with open(self.train_file,'r',encoding='utf-8',errors='replace') as f:
            data = f.readlines()

        res = []
        re_labels = set()  # 集合格式去重
        for did, d in enumerate(tqdm(data)):
            d = eval(d)
            text = d['text']
            gzsbs = []  # 存储故障设备
            gzyys = []  # 存储故障原因
            sbj_obj = []  # 存储真实的故障设备-故障原因
            for rel_id, spo in enumerate(d['spo_list']):
                tmp = {}
                tmp['text'] = text
                tmp['labels'] = []
                h, t = spo['h'], spo['t']
                h_name, t_name = h['name'], t['name']
                relation = spo['relation']
                tmp_rel_id = str(did)+'_'+str(rel_id)  # text的id和spo_list下的id,即索引
                tmp['id'] = tmp_rel_id
                tmp['labels'] = [h_name, t_name, relation]
                re_labels.add(relation)
                res.append(tmp)
                if h_name not in gzsbs:
                    gzsbs.append(h_name)
                if t_name not in gzyys:
                    gzyys.append(t_name)
                sbj_obj.append((h_name,t_name))

            # 构造负样本
            # 如果不在sbj_obj里则视为没有关系
            # 即：对于一个文本样本，其中真实的关系只有特定的设备和原因组合，
            # 其它设备-原因组合会被标记为“没关系”并加入训练数据
            # 这种方法有效增加了训练数据中的负样本，提升模型对无关关系的判别能力，同时避免生成过多负样
            # 本导致数据不均衡
            tmp = {}
            tmp['text'] = text
            tmp['labels'] = []
            tmp['id'] = str(did)+'_'+'norel'
            # 检查故障设备和故障原因是否都有多个，确保有机会生成多种组合
            # 确保至少有 2 个故障设备和 2 个故障原因，这样才能构造多个设备-原因的组合,
            # 这一条件避免了在数据稀少时构造无意义的负样本
            if len(gzsbs)>1 and len(gzyys)>1:
                # 通过设置上限，控制生成的负样本数量，避免因负样本过多导致训练数据严重不平衡
                neg_total = 3  # 希望生成 3 个负样本
                neg_cur = 0  # 当前生成的负样本计数

                for gzsb in gzsbs:
                    # 通过打乱 gzyys 的顺序，使得故障原因的选择更随机，提高负样本的多样性
                    random.shuffle(gzyys)

                    # 对每对设备和原因组合，检查它是否在 sbj_obj 中（真实关系列表）
                    # 如果不在，则认为它们没有关系，将其标记为 "没关系"，并加入负样本集合
                    for gzyy in enumerate(gzyys):
                        # 假设：gzyys = ["熄火", "抖动", "过热"]，执行 enumerate(gzyys) 的效果是生成：
                        # [(0, "熄火"), (1, "抖动"), (2, "过热")]，那么gzyy[1] = "熄火"
                        if (gzsb,gzyy[1]) not in sbj_obj:
                            tmp['labels'] = [gzsb,gzyy[1],'没关系']
                            res.append(tmp)
                            neg_cur += 1
                        break  # 每次只生成一个原因对应的负样本

                    if neg_cur == neg_total:
                        break  # 达到目标数量后停止

        train_ratio = 0.92
        train_num = int(len(res)*train_ratio)
        train_data = res[:train_num]
        dev_data = res[train_num:]

        with open(self.data_path+'re_data/train.txt','w',encoding='utf-8') as f:
            f.write('\n'.join([json.dumps(d,ensure_ascii=False) for d in train_data]))

        with open(self.data_path+'re_data/dev.txt','w',encoding='utf-8') as f:
            f.write('\n'.join([json.dumps(d,ensure_ascii=False) for d in dev_data]))

        labels = list(re_labels)+['没关系']
        with open(self.data_path+'re_data/labels.txt','w',encoding='utf-8') as f:
            f.write('\n'.join(labels))

processGgreData = ProcessDgreData()
processGgreData.get_ner_data()
processGgreData.get_re_data()

