# -*- coding: utf-8 -*-

from gensim.models import Word2Vec


model = Word2Vec.load("word2vec.model")
print(model.wv['侯亮平'])

#第一个是最常用的，找出某一个词向量最相近的词集合
req_count = 5
for key in model.wv.similar_by_word('沙瑞金', topn =100):
    if len(key[0])==3:
        req_count -= 1
        print(key[0], key[1])
        if req_count == 0:
            break;
            
#第二个应用是看两个词向量的相近程度，这里给出了书中两组人的相似程度
print (model.wv.similarity('沙瑞金', '高育良'))
print (model.wv.similarity('李达康', '王大路'))

#第三个应用是找出不同类的词，这里给出了人物分类题
print (model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split()))