# -*- coding: utf-8 -*-

# import modules & set up logging
import logging
import os
from gensim.models import word2vec


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt') 

#for sen in sentences:
#    print(sen)

model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
model.wv.save_word2vec_format("word2vec.bin", binary=True)   # 以二进制类型保存模型以便重用
model.wv.save_word2vec_format('word2vec.txt', binary=False)


#model.save("./word2vec.model")

#print(model.wv['侯亮平'])