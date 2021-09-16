import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from instrument.yuchuli import all_data2fold,Vocab,get_examples,batch2tensor,get_exampless,batch2tensorr
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from instrument.yuchuli import encode
test = pd.read_csv('test_a.csv')
data_file = r'C:\Users\Administrator\Desktop\Work\machine-learning\2021_7_20\tianchi\news_classification\train_set.csv'
# train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
# train_df[['text','label_ft']].to_csv('train.csv', index=None, header=None, sep='\t')

gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
device1 = torch.device('cpu')
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    device1 = torch.device('cpu')
else:
    device = torch.device("cpu")
n=5
# 将数据尽量类别均衡地分为n折数据
fold_data = all_data2fold(data_file, n, 20000)

train_labels = []
train_texts = []
lene = []



lens = [[len(i) for i in fold_data[j]['text']] for j in range(n)]

for i in range(n-1):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])
    lene.extend(lens[i])
sdim = 40000
nnlen = [i if i <=sdim else sdim for i in lene]
nnlenn = [i if i <=sdim else sdim for i in lens[n-1]]

train = {'label': train_labels, 'text': train_texts,'lens':nnlen}
test = {'text':fold_data[n-1]['text'],'label': fold_data[n-1]['label'],'lens':nnlenn}


# 学习词典
encodee = encode(train['text'],train['label'])
temp1 = encodee.code2label(train['label'])
temp1 = torch.tensor(temp1)
temp2 = encodee.code(train['text'])
temp2 = [torch.tensor(i) for i in temp2]
temp2 = pad_sequence(temp2,batch_first=True)
temp2 = [i[:sdim] for i in temp2]
temp2 = torch.stack(temp2,0)
train = {'label': temp1, 'text': temp2,'lens':lene}

temp1 = encodee.code2label(fold_data[n-1]['label'])
temp1 = torch.tensor(temp1)
temp2 = encodee.code(fold_data[n-1]['text'])
temp2 = [torch.tensor(i) for i in temp2]
temp2 = pad_sequence(temp2,batch_first=True)
temp2 = [i[:sdim] for i in temp2]
temp2 = torch.stack(temp2,0)
test = {'text':temp2,'label': temp1,'lens':lens[n-1]}


class mydata(Dataset):
    def __init__(self,train_encode,train_y,len):
        super(mydata,self).__init__()
        self.train_encode=train_encode
        self.train_y=train_y
        self.len = len
    def __len__(self):
        return len(self.train_encode)
    def __getitem__(self,idx):
        return self.train_encode[idx],self.train_y[idx],self.len[idx]

batch_size=20
# mydata具有自动将list转tensor的功能
dataset=mydata(train['text'],train['label'],train['lens'])
dataloader=DataLoader(dataset,batch_size=batch_size)
from torch.nn.modules import Module
class cbow(Module):
    def __init__(self, vocab_size, d_model, class_num):
        super(cbow, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.class_num = class_num
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.linear = nn.Linear(d_model, class_num)

    def forward(self, x, lens):
        # x[batch,maxlen],lens[batch]
        x = self.embed(x)  # x[batch,maxlen,d_model]
        x = x.sum(dim=1)  # x[batch,d_model]
        lens.unsqueeze_(1)
        x = x/lens

        output = self.linear(x)
        return output

vocab_size = 6038
d_model = 100
model=cbow(vocab_size,d_model,14).to(device)
model.train()
from torch.optim.adagrad import Adagrad

epochs=3000
lr=0.01
# cuda=torch.cuda.is_available()
# if(cuda):
#     model=model.cuda()
optimize=Adagrad(model.parameters(),lr)
lossCul=nn.CrossEntropyLoss()

for epoch in range(epochs):
    # 总损失
    allloss = 0
    for step, (x, y, z) in enumerate(dataloader):
        y.squeeze_()
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        output = model(x.long(),z)
        # 计算极大似然函数损失
        loss = lossCul(output, y)

        optimize.zero_grad()
        loss.backward()
        optimize.step()

        allloss += loss

    print("epochs:", epoch + 1, " iter:", step + 1, " loss:", allloss / (step + 1))
        # 验证集进行验证

model.eval()

with torch.no_grad():
            datass = test['text']
            labelss = test['label']
            model.to(device1)
            datass = datass.to(device1)
            leng = torch.tensor(test['lens'])
            outputt = model(datass.long(),leng)
            outputt = outputt.argmax(dim=1)
            # 计算准确率（因为其样本类别均匀）
            acc = int(sum(outputt == labelss)) / len(labelss)
            print("epochs:", epoch + 1, " iter:", step + 1, " acc:", acc)


