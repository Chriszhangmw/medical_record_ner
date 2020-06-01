
from config import Config
from apps.ner.albert_ner import AlbertNER,sess
from apps.utils.ourTokenizer import OurTokenizer
from bert4keras.utils import load_vocab
import numpy as np
from collections import  defaultdict

class ExtConbine(object):

    def __init__(self):
        self.models = []

        config_one = Config("v1","one")
        tokens = load_vocab(config_one.ALBERT_VOCAB_PATH)
        self.tokenizer = OurTokenizer(tokens)
        model1 = AlbertNER(config_one,self.tokenizer)

        config_two = Config("v1","two")
        model2 = AlbertNER(config_two,self.tokenizer)

        self.models.append(model1)
        self.models.append(model2)


    def predict(self,batch):
        X1,X2 = [],[]
        for text in batch:
            x1,x2 = self.tokenizer.encode(first_text = text.lower())
            X1.append(x1)
            X2.append(x2)
        X1 = self.seq_padding(X1)
        X2 = self.seq_padding(X2)

        tag_dict = []
        result = []
        for model in self.models:
            tags = model.predictrif(X1,X2)
            text_tags = self.decode(batch,tags)
            tag_dict.append(text_tags)
        for i in range(len(batch)):
            tmp_dict = {}
            for item in tag_dict:
                tmp_dict.updata(item[i])
            result.append(tmp_dict)
        return result

    def decode(self,batch,tags):
        batch_result = []
        for i in range(len(batch)):
            entities_dict = defaultdict(set)
            inputs_tags_list = list(zip(batch[i],tags[i]))
            for index,(s,t) in enumerate(inputs_tags_list):
                if len(t) > 1:
                    head,tag = t.split('-')
                    if head == "B":
                        tmp_str = s
                        for index1,(s1,t1) in enumerate(inputs_tags_list[index+1:]):
                            if t1 == "I-" + tag:
                                tmp_str += s1
                            else:
                                break
                        entities_dict[tag].add(tmp_str)
            result = {key:list(value) for key,value in entities_dict.items()}
            batch_result.append(result)
        return batch_result

    def ext_all(self,batch):
        tags_all = self.predict(batch)
        batch_result = self.decode(batch,tags_all)
        return batch_result

    def seq_padding(self,X,padding=0):
        L = [len(x) for x in X]
        ML = min(max(L),128)
        return np.array([np.concatenate([x,[padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X])





