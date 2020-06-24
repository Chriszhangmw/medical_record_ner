# medical_record_ner
albert for relation
eletrac for NER


1. 通过train_spo下面跑病例发现，在MRC结合start,end方式标注样本，训练样本，在处理较长文本的NER效果不佳。
比如一个长度300左右的病例（尽管训练的时候设置max length=512）,在病例前面的NER识别没问题，后面的NER就会识别错
所以初步推断，CRF的泛化能力比start，end的MRC方式要更强
