


import os
import json

#获取当前文件绝对目录的上级目录，即项目目录（config.py文件默认存放在项目目录下）
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

class Config:

    is_develop = json.load(open(PROJECT_PATH + '/branch.json'))["is_develop"]

    if is_develop:
        '''
        发布环境
        '''
        ALBERT_CONFIG_PATH = "/home/ai/pre_models/albert_samll_zh_google/albert_config_samll_google.json"
        ALBERT_CHECKPOINT_PATH = "/home/ai/pre_models/albert_samll_zh_google/albert_model.ckpt"
        ALBERT_VOCAB_PATH = "/home/ai/pre_models/albert_samll_zh_google/vocab.txt"

        BERT_CONFIG_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/bert_config.json"
        BERT_CHECKPOINT_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        BERT_VOCAB_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/vocab.txt"

        ELECTRA_CONFIG_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
        ELECTRA_CHECKPOINT_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
        ELECTRA_VOCAB_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"

    else:
        '''
        开发环境
        '''
        ALBERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_config_samll_google.json"
        ALBERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_model.ckpt"
        ALBERT_VOCAB_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/vocab.txt"

        BERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_config.json"
        BERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        BERT_VOCAB_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/vocab.txt"

        ELECTRA_CONFIG_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
        ELECTRA_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
        ELECTRA_VOCAB_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"

        #训练集
        DATA_TRAIN_PATH = os.path.join(PROJECT_PATH,"data/train_1224.json")
        DATA_DEV_PATH = os.path.join(PROJECT_PATH,"data/dev_1224.json")

    NER_DICT = {
        "v1":{
            "all":{"label_mapping_path":'apps/static/v1/label_mapping.json',
                   "model_path":"models/v1/bert_all_0525.h5"},
            "one":{
                "label_mapping_path":"apps/static/v1/label_mapping_5_1224.json",
                "model_path":"models/v1/bert_5_0426.h5"
            },

        }
    }

    desensitize_model_path = "models/v1/desensitize.h5"

    def __init__(self,version="v1",ner_type="one"):
        self.NER_MODEL_PATH = os.path.join(PROJECT_PATH,self.NER_DICT[version][ner_type]['model_path'])
        self.NER_MAPPING_PATH = os.path.join(PROJECT_PATH,self.NER_DICT[version][ner_type]['label_mapping_path'])





