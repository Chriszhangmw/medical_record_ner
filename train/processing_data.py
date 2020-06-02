

import pandas as pd
import json
import os
import sys
import numpy as np
import codecs
from numpy.random import shuffle


class ProcessingData(object):

    def __init__(self,config):

        self.config = config
        self.input_reg = u', . 。 ： ？ ( ) 【 】 “ ” " " = + - * / ~ ! @ # $ % ^ & 0 1 2 3 4 5 6 7 8 9 a b c d e f g h i j k l'
        self.output_reg = u',.：？() 】“”""=+-*/~!@#$%^&0123456789abcdefghijkl'

        self.mapping_path = config.NER_MAPPING_PATH

    def format_text(self,input_text):
        reg = {ord(f):ord(t) for f,t in zip(self.input_reg,self.output_reg)}
        output_text = input_text.translate(reg)
        return output_text

    def make_labels(self,input_data_path,is_clean_text=False):

        read_data_from_csv = pd.read_csv(input_data_path)

        text_label_list = []
        for data_dict in read_data_from_csv.to_dict('raw'):
            text_data_str = data_dict['文档内容']
            if is_clean_text is True:
                text_data_str = self.format_text(text_data_str)
            label_list = json.loads(data_dict["标注记录"])

            temp_text_data_list = []
            for chr in text_data_str:
                temp_text_data_list.append(chr + '\tO')
            for label_data in label_list:
                try:
                    label_class = label_data["label_name"]
                    s_index = label_data["mark_start_index"]
                    e_index = label_data["mark_end_index"]

                    s_label = "B-" + label_class + ""
                    temp_text_data_list[s_index] = temp_text_data_list[s_index].replace("O",s_label)

                    e_label = "I-" + label_class + ""
                    lable_length = e_index - s_index
                    for i in range(lable_length):
                        temp_text_data_list[s_index + i] = temp_text_data_list[s_index + i].replace("O",e_label)
                except Exception as err:
                    pass


            j = 0
            for i in range(len(temp_text_data_list)):
                if temp_text_data_list[j] == " \tO":
                    temp_text_data_list.pop(j)
                else:
                    j += 1

            text_label_list.append(temp_text_data_list)
        return text_label_list

    def cut_sentence(self,text_lists):
        new_texts = []
        split_patten = ['.',';','?']
        for text_label_list in text_lists:
            text_label_list = [item for item in [item.replace('\n','').replace('\r','') for item in text_label_list] if len(item) > 2]
            if len(text_label_list) < 126:
                new_texts.append(text_label_list)
                continue
            tmp = []
            for c in text_label_list:
                if len(c.split('\t')[0]) != 1:
                    continue
                tmp = []
                if c.split('\t')[0] in split_patten:
                    new_texts.append(tmp)
                    tmp = []


        return new_texts

    def load_data(self):
        if not os.path.exists(self.config.DATA_TRAIN_PATH):
            train_data,dev_data = [],[]
            text_label_list = self.make_labels(self.config.DATA_ORIFIN_PATH,is_clean_text=True)

            cuted_text_label_list = self.cut_sentence(text_label_list)

            shuffle(cuted_text_label_list)
            all_num = len(cuted_text_label_list)
            dev_num = all_num // 10
            i = 0
            train_wt = open(self.config.DATA_TRAIN_PATH,'a',encoding='utf-8')
            dev_wt = open(self.config.DATA_DEV_PATH,'a',encoding='utf-8')

            for text_lable in cuted_text_label_list:
                i += 1
                try:
                    splited_record = np.array([item.split('\t') for item in text_lable])
                    text,label = splited_record[:,0],list(splited_record[:,1])

                    if not any([item != 'O' for item in label]):
                        continue
                    text = ''.join(text)
                    if len(text) < 10:
                        continue
                    assert len(text) == len(label)

                    if i < dev_num:
                        json.dump([text,label],dev_wt,ensure_ascii=False)
                        dev_wt.write('\n')
                        dev_data.append([text,label])
                    else:
                        json.dump([text, label], dev_wt, ensure_ascii=False)
                        train_wt.write('\n')
                        train_data.append([text, label])
                except Exception as err:
                    continue
            return train_data,dev_data
        else:
            train_data = [json.loads(line.strip('\n')) for line in codecs.open(self.config.DATA_TRAIN_PATH,'r',encoding='utf-8').readlines()]
            dev_data = [json.loads(line.strip('\n')) for line in codecs.open(self.config.DATA_DEV_PATH,'r',encoding='utf-8').readlines()]

            _,label2index = self.load_mapping()

            def filter_label(label):
                if not any([int(label2index.get(item,0)) for item in label]):
                    return False
                else:
                    return True
            train_data = [item for item in train_data if filter_label(item[1])]
            dev_data = [item for item in dev_data if filter_label(item[1])]

            return train_data,dev_data

    def load_mapping(self,labels=None):
        if not os.path.exists(self.mapping_path):
            if labels is None:
                sys.exit(0)
            label2index = {"O":0}
            for label in labels:
                label2index["B-" + label] = len(label2index)
                label2index["I-" + label] = len(label2index)

            index2label = {j:i for i,j in label2index.items()}
            json.dump([index2label,label2index],open(self.config.NER_MAPPING_PATH,'w',encoding='utf-8'),ensure_ascii=False,indent=4)

            return index2label,label2index
        else:
            index2label,label2index = json.load(open(self.config.NER_MAPPING_PATH,'r',encoding='utf-8'))
            return index2label,label2index


if __name__ == "__main__":
    from config import Config
    config = Config()
    processTool = ProcessingData(config)
    labels = ['症状','疾病','身体部位','状态','生理特征','周期','治疗方式','日期','指标'
        ,'检查方式','药品名称','药品用法','指标值','药品用量','医院','度量单位','频率','药品类别']
    a = processTool.load_mapping(labels)
    print(a)








