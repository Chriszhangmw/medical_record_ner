
'''
利用electra加上crf做ner
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
bert_layers = 12
learning_rate = 1e-5
crf_lr_multiplier = 1000

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


#标签映射
labels = ['PER','LOC','ORG']
id2label = dict(enumerate(labels))
label2id = {j:i for i,j in id2label.items()}
num_labels = len(labels) * 2 + 1

class data_generator(DataGenerator):
    def __iter__(self,random=False):
        batch_token_ids,batch_segment_ids,batch_labels = [],[],[]
        for is_end, item in self.sample(random):
            token_ids,labels = [tokenizer._token_start_id],[0]
            for w,l in item:
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += ['O'] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids,batch_segment_ids],batch_labels
                batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

model = build_transformer_model(config_path=ELECTRA_CONFIG_PATH,checkpoint_path=ELECTRA_CHECKPOINT_PATH,model='electra')

output_layer = 'Transformer-%s-FeedForward-Norm' % (11)
output = model.get_layer(output_layer).output
output = Dense(num_labels)(output)

CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)
output = CRF(output)
model = Model(model.input,output)

model.compile(optimizer=Adam(1e-5),loss=CRF.sparse_loss,metrics=[CRF.sparse_accuracy])

class NamedEntityRecognizer(ViterbiDecoder):
    def recognize(self,text):
        tokens = tokenizer.tokenize(text)
        while len(tokens) > 512:
            tokens.pop(-2)
        mapping = tokenizer.rematch(text,tokens)
        token_ids = tokenizer.token_to_id(tokens)
        segment_ids = [0] * len(token_ids)
        nodes = model.predict([[token_ids],[segment_ids]])[0]
        labels = self.decode(nodes)

        entities,starting = [],False
        for i,label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i],id2label[(label-1)//2]])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False
        return [(text[mapping[w[0]][0]:mapping[w[-1]][-1]+1],1) for w,l in entities]



NER = NamedEntityRecognizer(trans=K.eval(CRF.trans),starts=[0],ends=[0])

def evaluate(data):
    X,Y,Z = 1e-10,1e-10,1e-10
    for d in tqdm(data):
        text = ''.join(i[0] for i in d)
        R = set(NER.recognize(text))
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1,precision,recall = 2 * X /(Y + Z),X/Y,X/Z
    return f1,precision,recall

class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0
    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        f1, precision, recall = evaluate(valid_data)

        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid: f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.5f\n' % (f1,precision,recall,self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test: f1: %.4f, precision: %.4f, recall: %.4f\n' % (
            f1, precision, recall)
        )



if __name__ == "__main__":
    #training
    evaluator = Evaluate()
    train_generator = data_generator(train_data,batch_size)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

    #validating
    model.load_weights('./best_model.weights')
    trans = K.eval(CRF.trans)
    NER.trans = trans
    p,r,f = evaluate(valid_data)
    print(p,r,f)


    #predicting
    model.load_weights('./best_model.weights')
    text = "我们编程艺术酷酷酷酷酷酷"
    tag = NER.recognize(text)
    print(tag)








