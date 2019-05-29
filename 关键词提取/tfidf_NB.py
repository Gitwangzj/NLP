'''
python实现tf-idf和朴素贝叶斯
'''
import jieba
import numpy as np 
from collections import defaultdict

class Corpus(object):
    def __init__(self, data):
        self.tags = defaultdict(int)
        self.vocabs = set()
        self.docs = []
        
        self.build_vocab(data)
        self.v_l = len(self.vocabs) # 字典的大小
        self.d_l = len(self.docs)   # 文档数
    
    # 分词器
    def tokenizer(self, sent):
        return jieba.lcut(sent)

    # 构建字典，获取分类标记集
    def build_vocab(self, data):
        for (tag, doc) in data:
            words = self.tokenizer(doc)
            self.vocabs.update(words)
            self.tags[tag] += 1
            self.docs.append((tag, words))
        self.vocabs = list(self.vocabs)
    
    # 计算词袋模型
    def calc_bow(self):
        # shape为 [d_l, v_l]，每一行存放文档的词袋向量
        self.bow = np.zeros([self.d_l, self.v_l])
        for idx in range(self.d_l):
            for word in self.docs[idx][1]:
                if word in self.vocabs:
                    self.bow[idx, self.vocabs.index(word)] += 1
    
    # 计算tf-idf
    def calc_tfidf(self):
        # 先计算bow，再用bow来计算tf
        self.calc_bow()
        
        # 初始化tf、df、idf
        self.tf = np.zeros([self.d_l, self.v_l])
        self.idf = np.ones([1, self.v_l])
        self.tf_idf = np.ones([self.d_l, self.v_l])
        for idx in range(self.d_l):
            self.tf[idx] = self.bow[idx] /np.sum(self.bow[idx])
            for word in self.docs[idx]:
                if word in self.vocabs:
                    self.idf[0, self.vocabs.index(word)] += 1
        self.idf = np.log(float(self.d_l) / self.idf)
        self.tfidf = self.tf * self.idf
        
    # 计算输入的bow向量，words代表输入序列（已分词）
    def get_idx(self, words):
        bow = np.zeros([1, self.v_l])
        for word in words:
            if word in self.vocabs:
                bow[0, self.vocabs.index(word)] += 1
        return bow

# NB继承了语料类Corpus
class NBayes(Corpus):
    def __init__(self, data, kernel="tfidf"):
        super(NBayes, self).__init__(data)
    
        # kernel 代表使用哪种特征，默认是tfidf，赋其他值代表使用bow
        self.kernel = kernel
        self.y_prob = {} # p(y_i)
        self.c_prob = None # p(x|y_i) , 计算条件概率
        self.feature = None
    
    # 训练，主要计算 p(y_i)和条件概率 p(x|y_i)
    def train(self):
        if self.kernel == "tfidf":
            self.calc_tfidf()
            self.feature = self.tfidf
        else:
            self.calc_bow()
            self.feature = self.bow
    
        # 采用极大似然估计计算p(y)
        for tag in self.tags:
            self.y_prob[tag] = float(self.tags[tag])/ self.d_l

        # 计算条件概率 p(x|y_i)
        self.c_prob = np.zeros([len(self.tags), self.v_l])
        Z = np.zeros([len(self.tags), 1]) # 归一化参数

        for idx in range(self.d_l):
            # 获得类别标签id
            tid = list(self.tags.keys()).index(self.docs[idx][0])
            self.c_prob[tid] += self.feature[idx]
            Z[tid] = np.sum(self.c_prob[tid])

        self.c_prob /= Z  # 归一化
    
    # 解码部分，返回使得概率值最大的类别y
    def predict(self, inp):
        words = self.tokenizer(inp)
        idx = self.get_idx(words)

        tag, score = None, -1
        for (p_c, y) in zip(self.c_prob, self.y_prob):
            tmp = np.sum(idx * p_c * self.y_prob[y])

            if tmp > score:
                tag = y
                score = tmp
        return tag, 1.0 - score

#测试
if __name__=="__main__":
    trainSet = [("pos", "good job !"),
                    ("pos", "表现不错哦"), 
                    ("pos", "厉害咯"), 
                    ("pos", "做的很好啊"), 
                    ("pos", "做得不错继续努力"),
                    ("pos", "不错！点赞"),
                    ("neg", "太差了"), 
                    ("neg", "太糟糕了"), 
                    ("neg", "你做的一点都不好"), 
                    ("neg", "不行，重做"),
                    ("neg", "so bad"),
                    ("non", "一般般吧，还过的去"), 
                    ("non", "不算太好，也不算太差"), 
                    ("non", "继续努力吧")
                   ]
                   
    nb = NBayes(trainSet)
    nb.train()
    print(nb.predict("不错哦")) # ('pos', 0.9142857142857143)
    
    
    
    
    
    
    
    
    
    
    
    