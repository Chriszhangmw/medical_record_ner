


import numpy as np

class data_generator:
    def __init__(self,data,label2index,tokenizer,batch_size=8):
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.label2index = label2index
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return  self.steps

    def seq_padding(self,X,padding=0):
        L = [len(x) for x in X]
        ML = min(max(L),128)
        return np.array([np.concatenate([x,[padding]*(ML - len(x))]) if len(x) < ML else x[:ML] for x in X])

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1,X2,Y = [],[],[]
            for i in idxs:
                try:
                    text,label = self.data[i]
                    if not any([int(self.label2index.get(item,0)) for item in label]):
                        continue
                    x1,x2 = self.tokenizer.encode(first_text= text.lower())

                    y = [-2] + [int(self.label2index.get(item,0)) for item in label] + [-2]
                    X1.append(x1)
                    X2.append(x2)
                    Y.append(y)

                    if len(X1) == self.batch_size or i == idxs[-1]:
                        X1 = self.seq_padding(X1)
                        X2 = self.seq_padding(X2)
                        Y = self.seq_padding(Y,padding=-1)
                        Y = np.expand_dims(Y,2)
                        yield [X1,X2],Y
                        [X1,X2,Y] = [],[],[]
                except Exception as err:
                    print(err)






