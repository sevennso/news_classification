import pandas as pd
from sklearn.metrics import f1_score

# 转换为FastText需要的格式
train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\Work\machine-learning\2021_7_20\tianchi\news_classification\train_set.csv',sep='\t', encoding='UTF-8', nrows=200000)
test = pd.read_csv('test_a.csv')
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')

import fasttext
help(fasttext.train_supervised)
model = fasttext.train_supervised('train.csv', lr=1.0, wordNgrams=2,
                                  verbose=2, minCount=1, epoch=25, loss="hs")

val = [model.predict(x)[0][0].split('__')[-1] for x in test['text']]

result = pd.read_csv('test_a_sample_submit.csv')
result['label'] = val
result.to_csv('submit1.csv', index=False)
# print(val_pred)
print(f1_score(train_df['label'].values[-5000:].astype(str), val, average='macro'))