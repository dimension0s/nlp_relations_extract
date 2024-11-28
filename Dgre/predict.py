import torch
from model import ner_model,re_model
from device import device
from ner_data import id2label as ner_id2label
from re_data import id2label as re_id2label
from ner_data import tokenizer
from itertools import combinations

# 测试函数主要因素：原始文本，分词，模型，解码
class Predictor:
    def __init__(self,ner_model,re_model,device,ner_id2label,re_id2label,tokenizer):
        self.device = device
        self.ner_model = ner_model.to(self.device)
        self.re_model = re_model.to(self.device)
        self.ner_id2label = ner_id2label
        self.re_id2label = re_id2label
        self.tokenizer = tokenizer

    # 1.实体识别
    def ner_predict(self,text):
        self.ner_model.eval()
        inputs = self.tokenizer(text,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.ner_model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits,dim=-1).cpu().numpy()

        entities = {}
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        for i,token in enumerate(tokens):
            label = self.ner_id2label[predictions[0][i]]  # 获取预测标签
            if label != 'O':  # 过滤掉'O'标签，'O'代表无实体
                if label not in entities:
                    entities[label] = []
                # 保存实体位置,比如：{'故障设备': [('空调', 5, 6)],'故障原因': [('制冷效果差', 7, 11)]}
                entities[label].append((token,i))
        # 返回实体信息和位置
        return entities


    # 2.关系抽取
    def re_predict(self,text,entities):
        self.re_model.eval()
        relations = []
        # 对于每一对实体，构造输入并预测关系
        for (entity1_label,entity1_pos),(entity2_label, entity2_pos) in combinations(entities.items(), 2):
            # 举例：entity1, entity1_pos 解包后，entity1 将是 '故障设备'，entity1_pos将是 [('空调', 5, 6)]
            for entity1, _ in entity1_pos:
                for entity2, _ in entity2_pos:
                    # 构造输入样本，加入实体1和实体2作为上下文
                    text_input = f"[CLS] {entity1} [SEP] {entity2} [SEP] {text} [SEP]"
                    inputs = self.tokenizer(text_input,
                                            max_length=512,
                                            padding=True,
                                            truncation=True,
                                            return_tensors='pt'
                                            ).to(self.device)

                    with torch.no_grad():
                        outputs = self.re_model(**inputs)
                        logits = outputs.logits
                        prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]  # 获取预测标签

                    if prediction != self.re_id2label['没关系']:
                        relations.append((entity1, entity2, self.re_id2label[prediction]))

        return relations

    # 3.联合测试
    def joint_test(self):
        # 进行实体识别
        entities = self.ner_predict(text)
        # 如果没有实体识别到，返回空的关系
        if not entities:
            print('No entities recognized.')
            return [],[]
        # 打印识别到的实体
        print(f"实体>>>>>： {entities}")

        # 进行关系抽取
        relations = self.re_predict(text, entities)
        print(f"关系>>>>>： {relations}")

        return entities,relations

# 4.实例化测试
texts = [
            "492号汽车故障报告故障现象一辆车用户用水清洗发动机后，在正常行驶时突然产生铛铛异响，自行熄火",
            "故障现象：空调制冷效果差。",
            "原因分析：1、遥控器失效或数据丢失;2、ISU模块功能失效或工作不良;3、系统信号有干扰导致。处理方法、体会：1、检查该车发现，两把遥控器都不能工作，两把遥控器同时出现故障的可能几乎是不存在的，由此可以排除遥控器本身的故障。2、检查ISU的功能，受其控制的部分全部工作正常，排除了ISU系统出现故障的可能。3、怀疑是遥控器数据丢失，用诊断仪对系统进行重新匹配，发现遥控器匹配不能正常进行。此时拔掉ISU模块上的电源插头，使系统强制恢复出厂设置，再插上插头，发现系统恢复，可以进行遥控操作。但当车辆发动在熄火后，遥控又再次失效。4、查看线路图发现，在点火开关处安装有一钥匙行程开关，当钥匙插入在点火开关内，处于ON位时，该开关接通，向ISU发送一个信号，此时遥控器不能进行控制工作。当钥匙处于OFF位时，开关断开，遥控器恢复工作，可以对门锁进行控制。如果此开关出现故障，也会导致遥控器不能正常工作。同时该行程开关也控制天窗的自动回位功能。测试天窗发现不能自动回位。确认该开关出现故障",
            "原因分析：1、发动机点火系统不良;2、发动机系统油压不足;3、喷嘴故障;4、发动机缸压不足;5、水温传感器故障。",
        ]
predictor = Predictor(ner_model, re_model, device, ner_id2label, re_id2label, tokenizer)
for text in texts:
    entities,relations = predictor.joint_test(text)
    print("文本>>>>>：", text)
    print("=" * 50)








