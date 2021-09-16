import numpy as np
import pandas as pd
import torch
from collections import Counter
# 返回fold_num折{'label','text'}数据


def all_data2fold(data_file, fold_num, num):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8',nrows = num)
    texts = f['text'].tolist()
    labels = f['label'].tolist()

    total = len(labels)
    index = list(range(total))
    np.random.shuffle(index)
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    # 创建label2id的字典
    # 字典中各个label键值保存对应数据的编号列表
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)



    # 将每个标签对应的数据均分到12个fold中
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        other = len(data) - batch_size * fold_num
        pos = 0
        for i in range(fold_num):
            cur_batch_size = batch_size + 1 if i < other else batch_size
            batch_data = [data[pos + b] for b in range(cur_batch_size)]
            pos = pos+cur_batch_size
            all_index[i].extend(batch_data)



    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels
        assert batch_size == len(fold_labels)


        data = {'label': fold_labels, 'text': fold_texts}
        fold_data.append(data)
    if start<len(other_texts):
        fold_data[-1]['label'].extend(other_labels[start:])
        fold_data[-1]['text'].extend(other_texts[start:])

    return fold_data



class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        self._id2label = []
        self.target_names = []
        self.build_vocab(train_data)

        reverse = lambda x: dict(zip(x, range(len(x))))
        #创建词和 index 对应的字典
        self._word2id = reverse(self._id2word)
        #创建 label 和 index 对应的字典
        self._label2id = reverse(self._id2label)

    def build_vocab(self, data):
        self.word_counter = Counter()
        #计算每个词出现的次数
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1
        # 去掉频次小于 min_count = 5 的词，把词存到 _id2word
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)

        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育', 7: '财经',
                      8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}

        self.label_counter = Counter(data['label'])

        for label in range(len(self.label_counter)):
            self._id2label.append(label)
            self.target_names.append(label2name[label]) # 根据label数字取出对应的名字

    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            # 第一行分别是单词数量、词向量维度
            word_count, embedding_dim = int(items[0]), int(items[1])

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        # 下面的代码和 word2vec.txt 的结构有关
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0]) # 首先添加第一列的单词
            vector = np.array(values[1:], dtype='float64') # 然后添加后面 100 列的词向量
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1

        # unk 的词向量是所有词的平均
        embeddings[self.unk] = embeddings[self.unk] / word_count
        # 除以标准差干嘛？
        embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        # assert len(set(self._id2extword)) == len(self._id2extword)

        return embeddings

    # 根据单词得到 id
    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)
    # 根据单词得到 ext id
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)
    # 根据 label 得到 id
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)

    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label)

#  将文章划分为等长度的句子，并且每篇文章最多提取16句话
def sentence_split(text, vocab, max_sent_len=256, max_segment=16):
    words = text.strip().split()
    document_len = len(words)
    # 划分句子的索引，句子长度为 max_sent_len
    index = list(range(0, document_len, max_sent_len))
    index.append(document_len)

    segments = []
    for i in range(len(index) - 1):
        # 根据索引划分句子
        segment = words[index[i]: index[i + 1]]
        assert len(segment) > 0
        # 把出现太少的词替换为
        segment = [word if word in vocab._id2word else '<UNK>' for word in segment]
        # 添加 tuple:(句子长度，句子本身)
        segments.append([len(segment), segment])

    assert len(segments) > 0
    # 如果大于 max_segment 句话，则局数减少一半，返回一半的句子
    if len(segments) > max_segment:
        segment_ = int(max_segment / 2)
        return segments[:segment_] + segments[-segment_:]
    else:
        # 否则返回全部句子
        return segments

def get_examples(data: object, vocab, max_sent_len=256, max_segment=8) -> object:
    examples = []
    for text, label in zip(data['text'], data['label']):
        # label
        label_id = vocab.label2id(label)
        # sents_words: 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            # 把 word 转为 id
            word_ids = vocab.word2id(sent_words)
            # 把 word 转为 ext id
            doc.append([sent_len, word_ids])
        examples.append([label_id, len(doc), doc])
    return examples

# 针对不含label的测试集
def get_exampless(data: object, vocab, max_sent_len=256, max_segment=8) -> object:
    examples = []
    for text in data['text']:

        # sents_words: 是一个list，其中每个元素是 tuple：(句子长度，句子本身)
        sents_words = sentence_split(text, vocab, max_sent_len, max_segment)
        doc = []
        for sent_len, sent_words in sents_words:
            # 把 word 转为 id
            word_ids = vocab.word2id(sent_words)
            # 把 word 转为 ext id
            doc.append([sent_len, word_ids])
        examples.append([len(doc), doc])
    return examples

def batch2tensor(batch_data):
        '''
            [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
        '''
        batch_size = len(batch_data)
        doc_labels = []
        doc_lens = []
        doc_max_sent_len = []
        for doc_data in batch_data:
            doc_labels.append(doc_data[0])
            doc_lens.append(doc_data[1])
            sent_lens = [sent_data[0] for sent_data in doc_data[2]]
            # 取出这篇新闻中最长的句子长度（单词个数）
            max_sent_len = max(sent_lens)
            doc_max_sent_len.append(max_sent_len)

        # 取出文章中句子最大的值
        max_doc_len = max(doc_lens)
        # 取出这批 batch 数据中最长的句子长度（单词个数）
        max_sent_len = max(doc_max_sent_len)
        # 创建 数据
        batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
        batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)
        batch_labels = torch.LongTensor(doc_labels)

        for b in range(batch_size):
            for sent_idx in range(doc_lens[b]):
                # batch_data[b][2] 表示一个 list，是一篇文章中的句子
                sent_data = batch_data[b][2][sent_idx]  # sent_data 表示一个句子
                for word_idx in range(sent_data[0]):  # sent_data[0] 是句子长度(单词数量)
                    # sent_data[1] 表示 word_ids
                    batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                    # # sent_data[2] 表示 extword_ids
                    # mask 表示 哪个位置是有词，后面计算 attention 时，没有词的地方会被置为 0
                    batch_masks[b, sent_idx, word_idx] = 1
        return (batch_inputs1, batch_masks), batch_labels


def batch2tensorr(batch_data):
    '''
        [[label, doc_len, [[sent_len, [sent_id0, ...], [sent_id1, ...]], ...]]
    '''
    batch_size = len(batch_data)
    doc_labels = []
    doc_lens = []
    doc_max_sent_len = []
    for doc_data in batch_data:
        doc_lens.append(doc_data[0])
        sent_lens = [sent_data[0] for sent_data in doc_data[1]]
        # 取出这篇新闻中最长的句子长度（单词个数），受split影响最大值为256
        max_sent_len = max(sent_lens)
        doc_max_sent_len.append(max_sent_len)

    # 取出最长的句子数量
    max_doc_len = max(doc_lens)
    # 取出这批 batch 数据中最长的句子长度（单词个数）
    max_sent_len = max(doc_max_sent_len)
    # 创建 数据
    batch_inputs1 = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.int64)
    batch_masks = torch.zeros((batch_size, max_doc_len, max_sent_len), dtype=torch.float32)


    for b in range(batch_size):
        for sent_idx in range(doc_lens[b]):
            # batch_data[b][2] 表示一个 list，是一篇文章中的句子
            sent_data = batch_data[b][1][sent_idx]  # sent_data 表示一个句子
            for word_idx in range(sent_data[0]):  # sent_data[0] 是句子长度(单词数量)
                # sent_data[1] 表示 word_ids
                batch_inputs1[b, sent_idx, word_idx] = sent_data[1][word_idx]
                # # sent_data[2] 表示 extword_ids
                # mask 表示 哪个位置是有词，后面计算 attention 时，没有词的地方会被置为 0
                batch_masks[b, sent_idx, word_idx] = 1
    return (batch_inputs1, batch_masks)

def bach_epo(data,label,batch_size):
        np.random.shuffle(data)
        state = np.random.get_state()
        np.random.set_state(state)
        np.random.shuffle(label)
        li = []
        batch_num = int(np.ceil(len(data) / float(batch_size)))
        for i in range(batch_num):
            # 如果 i < batch_num - 1，那么大小为 batch_size，否则就是最后一批数据
            cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
            docs = [data[i * batch_size + b] for b in range(cur_batch_size)]
            labels = [label[i * batch_size + b] for b in range(cur_batch_size)]
            docs = torch.stack(docs,0)
            labels = torch.stack(labels,0)
            li.append((docs,labels))
        return li


# 创建词典并且对句子进行编码
class encode():
    def __init__(self, x,y):
    # 拼接所有句子，制作词典
      y = [str(i) for i in y]
      clear_d = (' '.join(x).split(' '))
      clear_dd = (' '.join(y).split(' '))
      freq = dict(Counter(clear_d))
      label_freq = dict(Counter(clear_dd))
      word_freq = {}
    # 统计词频高于10的
      for k, v in freq.items():
        if v >= 10:
             word_freq[k] = v
      self.idx2word = ["<pad>"] + list(word_freq.keys()) + ["<unk>"]
      self.word2idx = {w: idx for idx, w in enumerate(self.idx2word)}
      self.idx2label = list(label_freq.keys())
      self.label2idx = {w: idx for idx,w in enumerate(self.idx2label)}
      self.vocab_size = len(self.idx2word)
      label_freq = {}

    def code(self,xx):
      hh = []
      for i in range(len(xx)):
         #每个句子直接分词
         sentences=xx[i].split(" ")
         #存储当前句子的编码信息
         c=[]
         for word in sentences:

                c.append(self.word2idx.get(word,self.word2idx["<unk>"]))
         hh.append(c)
      return hh

    def code2label(self,yy):
      yy = [str(i) for i in yy]
      hh = []
      for i in range(len(yy)):
          hh.append(self.label2idx.get(yy[i]))
      return hh






