# -*- coding: utf-8 -*-

import codecs
import sys
import pandas as pd
import numpy as np
from collections import deque  
import pdb
'''
#关系表
relation2id = {}
with codecs.open('relation2id.txt','r','utf-8') as input_data:
    for line in input_data.readlines():
        relation2id[line.split()[0]] = int(line.split()[1])
    input_data.close()
    print(relation2id)



datas = deque()
labels = deque()
positionE1 = deque() #位置
positionE2 = deque()
count = [0,0,0,0,0,0,0,0,0,0,0,0]
total_data=0
with codecs.open('train.txt','r','utf-8') as tfc: 
    for lines in tfc:
        line = lines.split()
        if count[relation2id[line[2]]] <1500:
            sentence = []
            index1 = line[3].index(line[0])
            position1 = []
            index2 = line[3].index(line[1])
            position2 = []
            #计算每个字与实体的距离
            for i,word in enumerate(line[3]):
                sentence.append(word)
                position1.append(i-index1)  
                position2.append(i-index2)
                i+=1
            datas.append(sentence)
            labels.append(relation2id[line[2]])
            positionE1.append(position1)
            positionE2.append(position2)
        count[relation2id[line[2]]]+=1
        total_data+=1
        
print (total_data,len(datas))
# python3废除了compiler，用下面函数代替
    
#from compiler.ast import flatten
    
import collections
def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

all_words = flatten(datas) #单字符拆解
sr_allwords = pd.Series(all_words)
sr_allwords = sr_allwords.value_counts() #字符统计

set_words = sr_allwords.index
set_ids = range(1, len(set_words)+1)
word2id = pd.Series(set_ids, index=set_words) #建立word2id词典，把每个字都转换成id
id2word = pd.Series(set_words, index=set_ids)

word2id["BLANK"]=len(word2id)+1
word2id["UNKNOW"]=len(word2id)+1
id2word[len(id2word)+1]="BLANK"
id2word[len(id2word)+1]="UNKNOW"
#print "word2id",id2word

#最大句子长度
max_len = 50
def X_padding(words):
    """把 words 转为 id 形式，并自动补全位 max_len 长度。"""
    ids = []
    for i in words:
        if i in word2id:
            ids.append(word2id[i])
        else:
            ids.append(word2id["UNKNOW"])
    if len(ids) >= max_len: 
        return ids[:max_len]
    print('********************')
    aaa = ids
    print(aaa)
    ids.extend([word2id["BLANK"]]*(max_len-len(ids))) 
    print('********************')
    aaaa = ids
    print(aaaa)

    return ids
    
    
def pos(num):
    if num<-40:
        return 0
    if num>=-40 and num<=40:
        return num+40
    if num>40:
        return 80
def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:  
        return words[:max_len]
    words.extend([81]*(max_len-len(words))) 
    return words


df_data = pd.DataFrame({'words': datas, 'tags': labels,'positionE1':positionE1,'positionE2':positionE2}, index=range(len(datas)))

a = df_data['words'][0][0:45]#['《', '水', '与', '火', '的', '缠', '绵', '》', '《', '低', '头', '不', '见', '抬', '头', '见', '》', '《', '天', '剑', '群', '侠', '》', '小', '品', '陈', '佩', '斯', '与', '朱', '时', '茂', '1', '9', '8', '4', '年', '《', '吃', '面', '条', '》', '合', '作', '者', '：', '陈', '佩', '斯', '聽', '1', '9', '8', '5', '年', '《', '拍', '电', '影', '》', '合']
X_padding(a)

b = df_data['positionE1'][0] #[-29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
position_padding(b)


df_data['tags'] = df_data['tags']
df_data['positionE1'] = df_data['positionE1'].apply(position_padding)
df_data['positionE2'] = df_data['positionE2'].apply(position_padding)
'''

#最大句子长度
max_len = 50
def pos(num):
    if num<-40:
        return 0
    if num>=-40 and num<=40:
        return num+40
    if num>40:
        return 80
def position_padding(words):
    words = [pos(i) for i in words]
    if len(words) >= max_len:  
        return words[:max_len]
    words.extend([81]*(max_len-len(words)))
    words_bak = words
    print(words_bak)
    return words

b = [-16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
c = [-41,-40,-39,-38,-37,-36,-35,-34,-33,-32,-31,-30,-29, -28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32,33,34]
position_padding(b)
position_padding(c)







