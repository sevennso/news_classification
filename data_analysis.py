import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\Work\machine-learning\2021_7_20\tianchi\news_classification\train_set.csv', sep='\t')
train_df['text_length'] = train_df['text'].apply(lambda x:len(x))
print(train_df['text_length'].describe())
plt.hist(train_df['text_length'],bins=100)
plt.title('News_class_count')
plt.xlabel('Char-Count')
plt.show()
# print(train_df['text_length'].describe())
train_df['unique']=train_df['text'].apply(lambda x:' '.join(list(set(x.split(' ')))))
alllines = ' '.join(list(train_df['unique']))
word_count = Counter(alllines.split(' '))
print(word_count.items())
word_count = sorted(word_count.items(),reverse = True,key = lambda d:int(d[1]))
print(word_count[0])
print(word_count[1])
print(word_count[2])
