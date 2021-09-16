import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
train_df = pd.read_csv(r'C:\Users\Administrator\Desktop\Work\machine-learning\2021_7_20\tianchi\news_classification\train_set.csv', sep='\t',nrows=1000)
print(train_df['text'])
test = pd.read_csv('test_a.csv')
# train_df['text']= train_df['text'].apply(lambda x:re.sub('3750',' ',x))
# train_df['text']= train_df['text'].apply(lambda x:re.sub('900',' ',x))
# train_df['text']= train_df['text'].apply(lambda x:re.sub('648',' ',x))
# train_df['text']= train_df['text'].apply(lambda x:re.sub('  ',' ',x))
#
# test['text']= test['text'].apply(lambda x:re.sub('3750',' ',x))
# test['text']= test['text'].apply(lambda x:re.sub('900',' ',x))
# test['text']= test['text'].apply(lambda x:re.sub('648',' ',x))
# test['text']= test['text'].apply(lambda x:re.sub('  ',' ',x))
vectorizer = CountVectorizer(max_features=3000)
# train_test = vectorizer.fit_transform(train_df['text']).astype('float32')
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=3000)
train_test = tfidf.fit_transform(train_df['text'])
testt = tfidf.fit_transform(test['text'])
model = XGBClassifier(n_estimators =200,max_depth=2)

model.fit(train_test,train_df['label'])
val = model.predict(testt)
# print(f1_score(train_df['label'].values[-5000:],val,average='macro'))

result = pd.read_csv('test_a_sample_submit.csv')
result['label'] = val
result.to_csv('submit.csv', index=False)

# model = LGBMClassifier()
# model = CatBoostClassifier()
# model = RandomForestClassifier()
# param_dist = {
    # 'n_estimators':range(50,400,50),
    #  'max_depth':range(2,15,5),
    # 'learning_rate':np.linspace(0.01,2,5),
    # 'subsample':np.linspace(0.7,0.9,5)
# }

# 2,100

# grid = GridSearchCV(model,param_dist,cv=3,scoring='neg_log_loss')
# grid.fit(train_test,train_df['label'])
# print(grid.best_params_)
# for train_set,test_set in kf.split(train_test):
#
#     model.fit(train_test[train_set],train_df['label'][train_set])
#     val_pred = model.predict(train_test[test_set])
#     result = result+f1_score(train_df['label'].values[test_set],val_pred,average='macro')
# print(result/4)



