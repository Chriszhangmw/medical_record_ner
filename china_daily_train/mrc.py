'''
利用分类的办法，不用crf

'''
import os
import numpy as np
from bert4keras.backend import keras,K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding,DataGenerator,open,ViterbiDecoder
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

ELECTRA_CONFIG_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
ELECTRA_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
ELECTRA_VOCAB_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"



maxlen = 256
epochs = 10
batch_size = 32
bert_layer = 12
learning_rate = 1e-5
crf_lr_multiplier = 1000

model_path = './mrc_sigmod.weights'

query_mapping = json.load(open('./mrc_query_mapping.json','r',encoding='utf-8'))

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

#建立分词器
tokenizer = Tokenizer(ELECTRA_VOCAB_PATH,do_lower_case=True)

class data_generator(DataGenerator):
    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_start,batch_end = [],[],[],[]
        for is_end,item in self.sample(random):
            for k,v in query_mapping.items():
                query_token_ids,query_segment_ids = tokenizer.encode(v)
                token_ids = query_token_ids.copy()
                start = query_segment_ids.copy()
                end = query_segment_ids.copy()
                for w,l in item:
                    w_token_ids = tokenizer.encode(w)[0][1:-1]
                    if len(token_ids) + len(w_token_ids) < maxlen:
                        token_ids += w_token_ids
                        start_tmp = [0] * len(w_token_ids)
                        end_tmp = [0] * len(w_token_ids)
                        if l == k:
                            start_tmp[0] = end_tmp[-1] = 1
                        start += (start_tmp)
                        end += (end_tmp)
                    else:
                        break
                token_ids += [tokenizer._token_end_id]
                segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_token_ids))
                start += [0]
                end += [0]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_start.append(to_categorical(start,2))
                batch_end.append(to_categorical(end,2))

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_start = sequence_padding(batch_start)
                    batch_end = sequence_padding(batch_end)
                    yield [batch_token_ids,batch_segment_ids,batch_start,batch_end],None
                    batch_token_ids,batch_segment_ids,batch_start,batch_end = [],[],[],[]

bert_model = build_transformer_model(
    config_path=ELECTRA_CONFIG_PATH,
    checkpoint_path=ELECTRA_CHECKPOINT_PATH,
    model='electra'
)


mask = bert_model.input[1]

start_labels = Input(shape=(None,2),name="start-labels")
end_labels = Input(shape=(None,2),name="end-labels")

output_layers = 'Transformer-%s-FeedForward-Norm' % (bert_layer -1)
x = bert_model.get_layer(output_layers).output

start_output = Dense(2,activation='sigmod',name='start')(x)
end_output = Dense(2,activation='sigmod',name='end')(x)

start_output = Lambda(lambda x:x ** 2)(start_output)
end_output = Lambda(lambda x:x ** 2)(end_output)


start_model = Model(bert_model.input,start_output)
end_model = Model(bert_model.input,end_output)

model = Model(bert_model.input + [start_labels,end_labels],[start_output,end_output])
model.summary()

def focal_loss(logits,labels,mask,lambda_param=1.5):
    probs = K.softmax(logits,axis=-1)
    pos_probs = probs[:,:,1]
    prob_label_pos = tf.where(K.equal(labels,1),pos_probs,K.ones_like(pos_probs))
    prob_label_neg = tf.where(K.equal(labels,0),pos_probs,K.zeros_like(pos_probs))

    loss = K.pow(1. - prob_label_pos,lambda_param) * K.log(prob_label_pos + 1e-7) +\
           K.pow(prob_label_neg,lambda_param)*K.log(1. - prob_label_neg + 1e-7)
    loss = -loss * K.cast(mask,'float32')
    loss = K.sum(loss,axis=-1,keepdims=True)
    loss = K.mean(loss)
    return loss

start_loss = K.binary_crossentropy(start_labels,start_output)
start_loss = K.mean(start_loss,2)
start_loss = K.sum(start_loss * mask) / K.sum(mask)

end_loss = K.binary_crossentropy(end_labels,end_output)
end_loss = K.mean(end_loss,2)
end_loss = K.sum(end_loss * mask) / K.sum(mask)

loss = start_loss + end_loss
model.add_loss(loss)
model.compile(optimizer=Adam(learning_rate))



def extract(text):
    result = set()
    for k,v in query_mapping.items():
        text_tokens = tokenizer.tokenize(text)[1:]
        query_tokens = tokenizer.tokenize(v)
        while len(text_tokens) + len(query_tokens) > 512:
            text_tokens.pop(-2)

        token_ids = tokenizer.token_to_id(query_tokens)
        token_ids += tokenizer.token_to_id(text_tokens)
        segment_ids = [0] * len(query_tokens) + [1] * len(text_tokens)

        start_out = start_model.predict([[token_ids],[segment_ids]])[0][len(query_tokens):]
        end_out = end_model.predict([[token_ids],[segment_ids]])[0][len(query_tokens):]

        start = np.where(start_out > 0.6)
        end = np.where(end_out > 0.5)

        start = [i for i,j in zip(*start) if j == 1]
        end = [i for i,j in zip(*end) if j ==1]
        for s,e in zip(start,end):
            if e >= s:
                tmp_str = ''.join(text_tokens[s:e+1])
                result.add((tmp_str,k))
    return result



class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def evaluate(self,data):
        X, Y, Z = 1e-10, 1e-10, 1e-10
        for d in tqdm(data):
            text = ''.join(i[0] for i in d)
            R = extract(text)
            T = set([tuple(i) for i in d if i[1] != 'O'])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        return f1, precision, recall

    def on_epoch_end(self, epoch, logs=None):

        f1, precision, recall = self.evaluate(valid_data)

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid: f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.5f\n' % (f1,precision,recall,self.best_val_f1)
        )
        f1, precision, recall = self.evaluate(test_data)
        print(
            'test: f1: %.4f, precision: %.4f, recall: %.4f\n' % (
            f1, precision, recall)
        )


if __name__ == "__main__":
    evaluator = Evaluate()
    train_generator = data_generator(train_data,batch_size)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    model.load_weights(model_path)
    p,r,f = evaluator.evaluate(test_data)
    print(p,r,f)







