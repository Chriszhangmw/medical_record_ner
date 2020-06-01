

# import numpy as np
# from bert4keras.backend import keras, K
# from bert4keras.models import build_transformer_model
# from bert4keras.tokenizers import Tokenizer
# from bert4keras.optimizers import Adam
# from bert4keras.snippets import sequence_padding, DataGenerator
# from bert4keras.snippets import open, ViterbiDecoder
# from bert4keras.layers import ConditionalRandomField
# from keras.layers import Dense
# from keras.models import Model
from tqdm import tqdm
from collections import defaultdict
from bert4keras.utils import Tokenizer,load_vocab
from  config import Config
from bert4keras.bert import build_bert_model
from bert4keras.layers import *
from bert4keras.train import  PiecewiseLinearLearningRate,GradientAccumulation
from keras.optimizers import Adam
from keras_contrib import  losses
import  os
import sys
from keras.models import load_model
from keras_contrib.layers.crf import CRF,crf_loss,crf_viterbi_accurary
from app.utils.tools import  Tool
from keras.backend import set_session




os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class AlbertNER(object):
    def __init__(self,configuration,tokenizer=None):
        if tokenizer is None:
            tokens = load_vocab(configuration.ALBERT_VOCAB_PATH)
            self.tokenizer = OurTokenizer(tokens)
        else:
            self.tokenizer = tokenizer


        tool = Tool(configuration.NER_MAPPING_PATH)
        self.index2label,_ = tool.load_mapping()
        self.index2label = {int(i):j for i,j in self.index2label.items()}

        if not os.path.exists(configuration.NER_MODEL_PATH):
            sys.exit()


        custom_objects = {
            "CRF":CRF,
            "crf_loss":crf_loss,
            "crf_viterbi_accuracy":crf_viterbi_accurary
        }

        set_session(sess)
        self.model = load_model(configuration.NER_MODEL_PATH,custom_objects)

    def seq_padding(self,X,padding=0):
        L = [len(x) for x in X]
        ML = min(max(L),128)
        return np.array([np.concatenate([x,[padding] * (ML - len(x))]) if len(x) < ML else x[:ML] for x in X])

    def predictrif(self,X1,X2):
        raws = self.model.predict([X1,X2])
        tags_all = []
        for i in range(len(X1)):
            raw = raws[i,1:1 + len(X1[i]),:-2]
            result = [np.argmax(row) for row in raw]
            tags = [self.index2label[int(i)] for i in result]
            tags_all.append(tags)
        return tags_all















