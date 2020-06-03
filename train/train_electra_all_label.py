

import os
import tensorflow as tf
from bert4keras.layers import *
from config import Config
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import load_vocab,Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator,sequence_padding
from train.processing_data import ProcessingData
from bert4keras.layers import ConditionalRandomField
from tqdm import tqdm
from collections import  defaultdict
import pandas as pd
from pandas.core.frame import DataFrame
from keras.backend import set_session
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

Hparameters = {"batch_size":32,
               "optimizer":Adam,
               "learning_rate":1e-5,
               "epochs":10,
               "crf_lr_multiplier":1000,
               "bert_layers":12}




class OurTokenizer(Tokenizer):
    def _tokenize(self,text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R

class MyDataGenerator(DataGenerator):
    def __init__(self,tokenizer,label2index,data,batch_szie=32,buffer_size=None):
        self.tokenizer = tokenizer
        self.label2index = label2index

        super(MyDataGenerator,self).__init__(data,batch_szie,buffer_size)

    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

        for is_end,item in self.sample(random):
            text,label = item
            x1,x2 = self.tokenizer.encode(first_text=text.lower())
            y = [0] + [int(self.label2index.get(item,0)) for item in label] + [0]

            batch_token_ids.append(x1)
            batch_segment_ids.append(x2)
            batch_labels.append(y)

            if len(batch_segment_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids,batch_segment_ids],batch_labels
                batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

config = Config(ner_type='all')
processTool = ProcessingData(config)

index2label,label2index = processTool.load_mapping()
index2label = {int(i):j for i,j in index2label.items()}


train_data,dev_data = processTool.load_data()
model_path = config.NER_MODEL_PATH



token_dict,keep_tokens = load_vocab(
    dict_path=Config.ELECTRA_CONFIG_PATH,
    simplified=True,
    startswith=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']
)



tokenizer = OurTokenizer(token_dict)

def create_model():
    bert_model = build_transformer_model(Config.ELECTRA_CONFIG_PATH,
                                         Config.ELECTRA_CHECKPOINT_PATH,
                                         model="electra")

    output_layer = 'Transformer-%s-FeedForward-Norm' % (12-1)
    output = bert_model.get_layer(output_layer).output
    output = keras.layers.Dense(len(label2index))(output)

    CRF = ConditionalRandomField(lr_multiplier=Hparameters['crf_lr_multiplier'])
    out = CRF(output)
    model = keras.models.Model(bert_model.input,out)
    model.compile(optimizer=Adam(1e-5),loss=CRF.sparse_loss,metrics=[CRF.sparse_accuracy])

    model.summary()


    return model,CRF

class NameEntityRecognizer():
    def __init__(self,model):
        self.model = model
        self.trans = K.eval(self.model.layers[-1].trans)

    def viterbi_decode(self,nodes,trans):
        labels = np.arange(len(index2label)).reshape((1,-1))
        scores = nodes[0].reshape((-1,1))
        scores[1:] -= np.inf
        paths = labels
        for l in range(1,len(nodes)):
            M = scores + trans + nodes[1].reshape((1,-1))
            idx = M.argmax(0)
            scores = M.max(0).reshape((-1,1))
            paths = np.concatenate([paths[:,idx],labels],0)

        return paths[:,scores[:,0].argmax()]

    def recognize(self,text):
        token_ids,segment_ids = tokenizer.encode(text)

        nodes = self.model.predict([[token_ids],[segment_ids]])[0]
        raw = self.viterbi_decode(nodes,self.trans)[1:-1]

        result_tags = [index2label[i] for i in raw]

        return result_tags


class Evaluate(keras.callbacks.Callback):
    def __init__(self,val_data,model,tokenizer,index2label,NER,CRF):
        self.model = model
        self.tokenizer = tokenizer
        self.F1 = []
        self.best = 0.
        self.val_data = val_data
        self.index2label = index2label
        self.labels = list(self.index2label.values())

        self.NER = NER
        self.CRF = CRF

    def on_epoch_end(self,epoch,logs=None):
        trans = K.eval(self.CRF.trans)
        self.NER.trans = trans

        [f1,precision,recall] = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            self.model.save(model_path)

        print('f1: %.4f, precision: %.4f, recall: %.4f, best F1: %.4f \n' % (f1,precision,recall,self.best))
        print('F1 list: ',self.F1)

    def evaluate(self):
        A,B,C = 1e-10,1e-10,1e-10
        for item in tqdm(self.val_data):
            x,y = item
            y = [item if item in self.labels else '0' for item in y]
            R = self.NER.recognize(x)
            T = y

            A += sum([1 for i in range(len(R)) if R[i] == T[i] and T[i] != '0'])
            B += len(R) - sum(i == '0' for i in R)
            C += len(T) - sum(i == '0' for i in T)

        return [2*A /(B + C), A/B, A/C]

def decode(inputs,tags):
    entities_dict = defaultdict(set)
    inputs_tags_list = list(zip(inputs,tags))

    for index,(s,t) in enumerate(inputs_tags_list):
        if len(t) > 1:
            head,tag = t.split('-')
            if head == 'B':
                tmp_str = s
                for index1,(s1,t1) in enumerate(inputs_tags_list[index+1:]):
                    if t1 == 'I-' + tag:
                        tmp_str += s1
                    else:
                        break
                entities_dict[tag].add(tmp_str)
    return  entities_dict


def test(test_datas):
    trans = K.eval(CRF.trans)
    NER.trans = trans
    metric = defaultdict(lambda:[1e-10,1e-10,1e-10])
    result = defaultdict(list)

    for item in tqdm(test_datas):
        x,y = item
        y = [item if item in label2index else 'O' for item in y]
        y_pre = NER.recognize(x)
        y_tags = decode(x,y)
        y_pre_tags = decode(x,y_pre)
        for key,value in y_tags.items():
            a = len(value)
            b = len(y_pre_tags[key])
            c = len(value & y_pre_tags[key])

            metric[key][0] += a
            metric[key][1] += b
            metric[key][2] += c

    for k,v in metric.items():
        A = v[0]
        B = v[1]
        C = v[2]

        P = C/A
        R = C/B

        F1 = 2 * C / (A + B)

        result[k].extend([P,R,F1])

    a = DataFrame(DataFrame(result).T.values,columns=["P","R","F1"],index=list(result.keys()))
    pd.set_option('display.max_columns',None)
    return result

def train():
    model,CRF = create_model()
    NER = NameEntityRecognizer(model)
    evalutor = Evaluate(dev_data,model,tokenizer,index2label,NER,CRF)

    train_D = MyDataGenerator(tokenizer,label2index,train_data,batch_szie=Hparameters['batch_size'])

    model.fit_generator(
        train_D.forfit(),
        steps_per_epoch=len(train_D),
        epochs=10,
        callbacks=[evalutor])

class test_sentence():
    def __init__(self):
        bert_model = build_transformer_model(Config.ELECTRA_CONFIG_PATH,
                                             Config.ELECTRA_CHECKPOINT_PATH,
                                             model="electra")
        output_layer = "Transformer-%s-FeedForward-Norm" % (12-1)
        output = bert_model.get_layer(output_layer).output
        output = keras.layers.Dense(len(label2index))(output)
        CRF = ConditionalRandomField(lr_multiplier=1000)
        out = CRF(output)
        model = keras.models.Model(bert_model.input,out)
        model.compile(optimizer=Adam(1e-5),loss=CRF.sparse_loss,metrics=[CRF.sparse_accuracy])
        set_session(sess)

        model.load_weights(model_path)

        self.NER = NameEntityRecognizer(model)

    def decode(self,inputs,tags):
        entities_dict = defaultdict(set)
        inputs_tags_list = list(zip(inputs,tags))
        for index,(s,t) in enumerate(inputs_tags_list):
            if len(t) > 1:
                head,tag = t.split('-')
                if head == "B":
                    tmp_str = s
                    for index1,(s1,t1) in enumerate(inputs_tags_list[index+1:]):
                        if t1 == 'I-' + tag:
                            tmp_str += s1
                        else:
                            break
                    entities_dict[tag].add((index,tmp_str))
        result = {key:list(value) for key,value in entities_dict.items()}
        return  result

    def predict(self,text):
        tags = self.NER.recognize(text)
        result = self.decode(text,tags)
        return result



if __name__ == "__main__":
  #train()
  test = test_sentence()
  print(test.predict("现病史：患者半年前出现胸闷气短，咳嗽，后来去省人民医院为新冠肺炎"))
























