

from bert4keras.utils import Tokenizer,load_vocab
import json

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

class TokenizerFactory(object):

    def __init__(self,all_data,vocab_path,is_ner=False):
        self.is_ner = is_ner
        self.vocab_path = vocab_path
        self.min_count = 2
        self.all_data = all_data

    def get_Tokenizer(self):
        tokens = {}
        token_dict,keep_words = {},[]
        _token_dict = load_vocab(self.vocab_path)
        if self.is_ner:
            _tokenizer = OurTokenizer(_token_dict)
        else:
            _tokenizer = Tokenizer(_token_dict)

        for text in self.all_data:
            for c in _tokenizer.tokenize(text):
                tokens[c] = tokens.get(c,0) + 1

        tokens = {i:j for i,j in tokens.items() if j >= self.min_count}
        json.dump(list(tokens.keys()),open('ner_tokens.json','w',encoding='utf-8'),ensure_ascii=False,indent=4)
        for t in ["[PAD]","[UNK]","[CLS]","[SEP]"]:
            token_dict[t] = len(token_dict)
            keep_words.append(_token_dict[t])

        for t in tokens:
            if t in _token_dict and t not in token_dict:
                token_dict[t] = len(token_dict)
                keep_words.append(_token_dict[t])
        if self.is_ner:
            tokenizer = OurTokenizer(token_dict)
        else:
            tokenizer = Tokenizer(token_dict)
        return tokenizer,keep_words

def make_token_me():
    import pandas as pd
    import codecs
    tokens = {}
    _token_dict = load_vocab(vocab_path)






