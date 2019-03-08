# -*- coding: UTF-8 -*-

import codecs
import re
import pdb
import pandas as pd
import numpy as np
import collections

'''
人民日报：只提取人名、地名、组织名三种实体类型
首先去掉语料的前缀
'''
def originHandle():
    with open('../data/renMinRiBao/renmin.txt','r',encoding='UTF-8') as inp,open('../data/renMinRiBao/renmin2.txt','w') as outp:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i<len(line)-1:
                if line[i][0]=='[':
                    outp.write(line[i].split('/')[0][1:])
                    i+=1
                    while i<len(line)-1 and line[i].find(']')==-1:
                        if line[i]!='':
                            outp.write(line[i].split('/')[0])
                        i+=1
                    outp.write(line[i].split('/')[0].strip()+'/'+line[i].split('/')[1][-2:]+' ')
                elif line[i].split('/')[1]=='nr':
                    word = line[i].split('/')[0] 
                    i+=1
                    if i<len(line)-1 and line[i].split('/')[1]=='nr':
                        outp.write(word+line[i].split('/')[0]+'/nr ')           
                    else:
                        outp.write(word+'/nr ')
                        continue
                else:
                    outp.write(line[i]+' ')
                i+=1
            outp.write('\n')
'''
实体标注：nt-组织名，nr-人名，ns-地名，o-单字，M-词中
'''
def originHandle2():
    with codecs.open('../data/renMinRiBao/renmin2.txt','r') as inp,codecs.open('../data/renMinRiBao/renmin3.txt','w','utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i<len(line)-1:
                if line[i]=='':
                    i+=1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag=='nr' or tag=='ns' or tag=='nt':
                    outp.write(word[0]+"/B_"+tag+" ")
                    for j in word[1:len(word)-1]:
                        if j!=' ':
                            outp.write(j+"/M_"+tag+" ")
                    outp.write(word[-1]+"/E_"+tag+" ")
                else:
                    for wor in word:
                        outp.write(wor+'/O ')
                i+=1
            outp.write('\n')
'''
去掉标点符号
'''           
def sentence2split():
    with open('../data/renMinRiBao/renmin3.txt','rb') as inp,codecs.open('../data/renMinRiBao/renmin4.txt','w') as outp:
        texts = inp.read().decode('utf-8')
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
	        if sentence != " ":
		        outp.write(sentence.strip()+'\n')     
'''
划分训练集和测试集，验证集
'''

def data2pkl():
    datas = list()
    labels = list()
    linedata=list()
    linelabel=list()
    tags = set()
    tags.add('')
    input_data = codecs.open('../data/renMinRiBao/renmin4.txt','r',encoding='gbk')
    for line in input_data.readlines():
        line = line.split()
        linedata=[]
        linelabel=[]
        numNotO=0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            tags.add(word[1])
            if word[1]!='O':
                numNotO+=1
        if numNotO!=0:
            datas.append(linedata)
            labels.append(linelabel)

    input_data.close()
    print (len(datas))
    print (len(labels))
    
    '''
    python3废除了compiler，用下面函数代替
    '''
#    from compiler.ast import flatten
    
    import collections
    def flatten(x):
        result = []
        for el in x:
            if isinstance(x, collections.Iterable) and not isinstance(el, str):
                result.extend(flatten(el))
            else:
                result.append(el)
        return result
    
    all_words = flatten(datas)
    
    sr_allwords = pd.Series(all_words)
    sr_allwords = sr_allwords.value_counts()
    set_words = sr_allwords.index
    set_ids = range(1, len(set_words)+1)

    
    tags = [i for i in tags]
    tag_ids = range(len(tags))
    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)
    word2id["unknow"]=len(word2id)+1
    id2word[len(word2id)]="unknow"
    print (tag2id)
    max_len = 60
    def X_padding(words):
        ids = list(word2id[words])
        if len(ids) >= max_len:  
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids

    def y_padding(tags):
        ids = list(tag2id[tags])
        if len(ids) >= max_len: 
            return ids[:max_len]
        ids.extend([0]*(max_len-len(ids))) 
        return ids
    df_data = pd.DataFrame({'words': datas, 'tags': labels}, index=range(len(datas)))
    df_data.to_csv('df_data.csv',index=False)
    df_data['x'] = df_data['words'].apply(X_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)
    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))
    df_data.to_csv('df_data_x_y.csv',index=False)
    from sklearn.model_selection import train_test_split
    x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train,  test_size=0.2, random_state=43)


    import pickle
    import os
    with open('../data/renMinRiBao/renmindata.pkl', 'wb') as outp:
	    pickle.dump(word2id, outp)
	    pickle.dump(id2word, outp)
	    pickle.dump(tag2id, outp)
	    pickle.dump(id2tag, outp)
	    pickle.dump(x_train, outp)
	    pickle.dump(y_train, outp)
	    pickle.dump(x_test, outp)
	    pickle.dump(y_test, outp)
	    pickle.dump(x_valid, outp)
	    pickle.dump(y_valid, outp)
    print ('** Finished saving the data.')
    
##################Main#####################
#originHandle()
#originHandle2()
#sentence2split()
data2pkl()
