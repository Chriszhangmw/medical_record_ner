




import os,sys,json

class Tool(object):
    def __init__(self,mapping_path):
        self.mapping_path = mapping_path

    def load_mapping(self,labels=None):
        if not os.path.exists(self.mapping_path):
            if labels is None:
                sys.exit(0)
            label2index = {"0":0}
            for label in labels:
                label2index["B-"  + label] = len(label2index)
                label2index["I-" + label] = len(label2index)

            index2label = {j:i for i,j in label2index.items()}
            json.dump([index2label,label2index],open(self.mapping_path,'w',encoding='utf-8'),ensure_ascii=False,indent=4)

            return index2label,label2index
        else:
            index2label,label2index = json.load(open(self.mapping_path,'r',encoding='utf-8'))
            return index2label,label2index








