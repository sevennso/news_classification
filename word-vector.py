import jieba
import gensim
from gensim.models import word2vec
import logging
tt ='0324 1234 5467 8972 9980 7750 2567'


with open('1120.txt', errors='ignore', encoding='utf-8') as fp:
   lines = fp.readlines()
   for line in lines:
       seg_list = jieba.cut(line)
       with open('1130.txt', 'a', encoding='utf-8') as ff:
           ff.write(' '.join(seg_list)) # 词汇用空格分开

from gensim.models import word2vec

# 加载语料
sentences = word2vec.Text8Corpus('1130.txt')

# 训练模型
model = word2vec.Word2Vec(sentences,min_count=1)

# 选出最相似的10个词
for e in model.wv.most_similar(positive=['华生'], topn=10):
   print(e[0], e[1])

print(model.wv['福尔摩斯'])
print(model.wv.similarity('华生','福尔摩斯'))
print(model.wv.similarity('希尔达','福尔摩斯'))
print(model.wv.similarity('华生','医生'))
print(model.wv.similarity('福尔摩斯','医生'))