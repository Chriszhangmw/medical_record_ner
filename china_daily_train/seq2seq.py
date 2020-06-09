


import os
import numpy as np
from bert4keras.backend import keras,K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer,load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding,DataGenerator,open,ViterbiDecoder,AutoRegressiveDecoder
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import json
import tensorflow as tf
from keras.layers import Input,Lambda
from keras.utils import to_categorical
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ALBERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_config_samll_google.json"
ALBERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_model.ckpt"
ALBERT_VOCAB_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/vocab.txt"

BERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_config.json"
BERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
BERT_VOCAB_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/vocab.txt"

#用electra会报错这里
ELECTRA_CONFIG_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
ELECTRA_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
ELECTRA_VOCAB_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"

batch_size = 32
max_len = 256

model_path = 'best_ner.h5'

def load_data(filename):
    D = []
    with open(filename,encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d,last_flag = [],''
            for c in l.split('\n'):
                char,this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char,'O'])
                elif this_flag[:1] == 'B':
                    d.append([char,this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D




#标注数据
train_data = load_data('./datasets/example.train')
valid_data = load_data('./datasets/example.dev')
test_data = load_data('./datasets/example.test')


query_mapping = json.load(open('./mrc_query_mapping.json','r',encoding='utf-8'))


#加载并精简词表，建立分词器
token_dict,keep_tokens = load_vocab(
    dict_path=BERT_VOCAB_PATH,
    simplified=True,
    startswith=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']
)

tokenizer = Tokenizer(token_dict)

class MyDataGenerator(DataGenerator):
    def __init__(self,data,batch_size = 32, buffer_size=None):
        super(MyDataGenerator,self).__init__(data,batch_size,buffer_size)

    def __iter__(self,random=False):
        batch_token_ids, batch_segment_ids = [],[]
        for is_end, item in self.sample(random):
            '''
            单条样本：[[cls],query,[sep],text,[sep],entity1#entity2...[sep]]
            '''
            for k,v in query_mapping.items():
                query_token_ids, query_segment_ids = tokenizer.encode(v)
                token_ids = query_token_ids.copy()
                entity_ids = []
                for w,l in item:
                    w_token_ids = tokenizer.encode(w)[0][1:-1]
                    if l == k:
                        entity_ids.extend(w_token_ids + [tokenizer.token_to_id('@')])
                    if len(token_ids) + len(w_token_ids) + len(entity_ids) < max_len:
                        token_ids += w_token_ids
                    else:
                        break
                token_ids += [tokenizer._token_end_id]
                segment_ids = [0] * len(token_ids) + [1] * (len(entity_ids) + 1)

                token_ids += entity_ids
                token_ids += [tokenizer._token_end_id]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    yield [batch_token_ids,batch_segment_ids],None
                    batch_token_ids,batch_segment_ids = [],[]

model = build_transformer_model(
    config_path=BERT_CONFIG_PATH,
    checkpoint_path= BERT_CHECKPOINT_PATH,
    application='unilm',
    keep_tokens=keep_tokens
)

y_true = model.input[0][:,1:]
y_mask = model.input[1][:,1:]
y_pred = model.output[:,:-1]


cross_entropy = K.sparse_categorical_crossentropy(y_true,y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(learning_rate=1e-5))
model.summary()


class AutoAnswer(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids,output_ids],1)
        segment_ids = np.concatenate([segment_ids,np.ones_like(output_ids)],1)

        return model.predict([token_ids,segment_ids])[:,-1]


    def generate(self,text,topk=1):
        result = set()
        for k,v in query_mapping.items():
            token_ids,_ = tokenizer.encode(first_text=v,second_text=text)
            segment_ids = [0] * len(token_ids)
            output_ids = self.beam_search([token_ids,segment_ids],topk)
            entity_str = tokenizer.decode(output_ids)
            entity = [item.strip() for item in entity_str.split('@')] if entity_str else []
            for item in entity:
                if item:
                    result.add((item,k))
        return  result

auto_answer = AutoAnswer(start_id=None,end_id=tokenizer._token_end_id,maxlen=64)

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def evaluate(self,data):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in tqdm(data):
            text = ''.join(i[0] for i in d)
            if len(text) > max_len:
                continue
            T = set([tuple(i) for i in d if i[1] != 'O'])
            R = auto_answer.generate(text)
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs=None):

        f1, precision, recall = self.evaluate(valid_data)

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save(model_path)
        print(
            'valid: f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.5f\n' % (f1,precision,recall,self.best_val_f1)
        )




if __name__ == "__main__":
    train_d = MyDataGenerator(train_data,batch_size)
    evalutor = Evaluate()
    model.fit_generator(train_d.forfit(),epochs=10,steps_per_epoch=len(train_d),callbacks=[evalutor])















