import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from instrument.yuchuli import bach_epo
from collections import Counter
from instrument.yuchuli import all_data2fold,Vocab,get_examples,batch2tensor,get_exampless,batch2tensorr
import numpy as np

test = pd.read_csv('test_a.csv')
data_file = r'C:\Users\Administrator\Desktop\Work\machine-learning\2021_7_20\tianchi\news_classification\train_set.csv'


gpu = 0
use_cuda = gpu >= 0 and torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    device1 = torch.device('cpu')
else:
    device = torch.device("cpu")

# 将数据尽量类别均衡地分为n折数据
fold_data = all_data2fold(data_file, 4, 20000)

train_labels = []
train_texts = []

for i in range(3):
    data = fold_data[i]
    train_texts.extend(data['text'])
    train_labels.extend(data['label'])
train_data = {'label': train_labels, 'text': train_texts}
test = {'text':fold_data[3]['text'],'label': fold_data[3]['label']}

# 创建词典
vocab = Vocab(train_data)
aaa = get_examples(train_data,vocab)
# bbb = get_exampless(test,vocab)
bbb = get_examples(test,vocab)

datas,labels = batch2tensor(aaa)
# datass = batch2tensorr(bbb)
datass,labelss = batch2tensor(bbb)

datass = datass[0].view(-1,2048)
datass  = datass.to(device)
labelss = labelss.to(device)
sdim = 2048
mm = datas[0].view(-1,sdim)
nnn = datass[0].view(-1,2048)

# nn = datass[0].view(-1,2048)


# train_dataa = mm.to(device)
# test_data = nnn.to(device)
# train_labell = torch.LongTensor(train_data['label']).to(device)
# test_labell = torch.LongTensor(fold_data[4]['label']).to(device)
# test_data = nn[:,:va].to(device)

# test_dataa = dataa[4000:].to(device)
# test_labell = torch.LongTensor(train_data['label'][4000:]).to(device)




class FastText(nn.Module):
    def __init__(self, vocab, w2v_dim, classes, hidden_size):
        super(FastText, self).__init__()
        #创建embedding
        self.embed = nn.Embedding(vocab, w2v_dim)  #embedding初始化，需要两个参数，词典大小、词向量维度大小
        self.embed.weight.requires_grad = True #需要计算梯度，即embedding层需要被训练
        self.fc = nn.Sequential(              #序列函数
            nn.Linear(sdim*w2v_dim, hidden_size),  #这里的意思是先经过一个线性转换层
            nn.BatchNorm1d(100),      #再进入一个BatchNorm1d
            nn.ReLU(inplace=True),            #再经过Relu激活函数
            nn.Linear(hidden_size, classes)#最后再经过一个线性变换
        )

    def forward(self, x):
        x = self.embed(x)                     #先将词id转换为对应的词向量
        x = x.view(-1,2048*w2v_dim)
        out = self.fc(x)   #这使用torch.mean()将向量进行平均
        return out

def train_model(net, epoch, lr, data, label, batch_size):      #训练模型
    print("begin training")
    net.train()  # 将模型设置为训练模式，很重要！
    optimizer = optim.Adam(net.parameters(), lr=lr) #设置优化函数
    Loss = nn.CrossEntropyLoss()  #设置损失函数
    qq = bach_epo(data,label,batch_size)
    for i in range(epoch):  # 循环
      for m,n in qq:
        m=m.to(device)
        n=n.to(device)
        optimizer.zero_grad()  # 清除所有优化的梯度
        output = net(m)  # 传入数据，前向传播，得到预测结果
        loss = Loss(output, n) #计算预测值和真实值之间的差异，得到loss
        loss.backward() #loss反向传播
        optimizer.step() #优化器优化参数

        # 打印状态信息
      print("train epoch=" + str(i) + ",loss=" + str(loss.item()))
    print('Finished Training')

def model_test(net, test_data, test_label):
    net.eval()  # 将模型设置为验证模式
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = net(test_data)
        # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
        _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
        print(predicted[:20])
        total += test_label.size(0)
        correct += (predicted == test_label).sum().item()
        print('Accuracy: %d %%' % (100 * correct / total))



if __name__ == "__main__":
    #这里没有写具体数据的处理方法，毕竟大家所做的任务不一样

    epoch = 100  # 迭代次数
    w2v_dim = 100  # 词向量维度
    lr = 0.001
    hidden_size = 100
    classes = 14
    vocabb = 6038
    batch_size = 100

    # 定义模型
    net = FastText(vocab=vocabb, w2v_dim=w2v_dim, classes=classes, hidden_size=hidden_size).to(device)

    # 训练
    print("开始训练模型")
    train_model(net, epoch, lr, mm, labels, batch_size)
    # 保存模型
    print("开始测试模型")
    # print(test_labell.shape)
    model_test(net, datass, labelss)
    # net.eval()
    # resultt = torch.tensor([]).to(device)
    # with torch.no_grad():
    #  for i in range(10):
    #    nnnn = nnn[i*5000:i*5000+5000].to(device)
    #    result = net(nnnn)
    #    _, predicted = torch.max(result.data, 1)
    #    print(predicted)
    #    resultt = torch.cat((resultt,predicted))
    # result = pd.read_csv('test_a_sample_submit.csv')
    # result['label'] = resultt
    # result.to_csv('submit1.csv', index=False)
    # print(resultt[:20])