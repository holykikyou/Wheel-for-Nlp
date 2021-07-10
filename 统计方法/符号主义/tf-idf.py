"""
2021-03
简单实现
输入已经分好词的句子列表即二维词矩阵
输出词权重矩阵
"""

import numpy as np
from collections import Counter
import time
class myTF_IDF():
    """注意计算速度"""
    def __init__(self,sentence_list:list):
        # self.sentence_list=sentence_list
        self.countlist=self.count_freq(sentence_list)

    def count_freq(self,sentence_list):
        """返回每段话中的词频列表"""
        count_list=[]
        for word_list in sentence_list:
            count_list.append(Counter(word_list))
        return count_list
    def count_tf(self,word,count):
        """count是句子的词频词典，计算词在句子里的tf"""
        return count[word]/sum(count.values())
    def count_idf(self,word,countlist):
        """注意计算n的代码，分子加一是为了防止"""
        n=sum([1 for count in countlist if word in count ])  #包含word的count

        return (np.log(len(countlist))+1/n)
    def count_tf_idf(self,word,count,countlist):
        """说明tfidf值既跟所有语料有关，也跟它所在句子有关"""
        tf_idf=self.count_tf(word,count)*self.count_idf(word,countlist)
        return tf_idf

    def print_tfidf_score(self):
        """非常低效的方式，应该维护一个idf表，而不是每次计算tfid时都要重新算一遍"""
        start_time = time.time()
        for i in range(len(self.countlist)):
            print(f"第{i}句:")
            for word in self.countlist[i]:
                print(word,' ',self.count_tf_idf(word,self.countlist[i],self.countlist),'\t')
        print('耗时：%d' % (time.time() - start_time))

    def more_efficent_print(self):
        start_time=time.time()
        word_idf_dict={}
        for count in self.countlist:
            for word in count.keys():
                word_idf_dict[word]=self.count_idf(word,self.countlist) #预先获得词集合同样要遍历一遍，差不多
        for i,count in enumerate(self.countlist):
            print(f"第{i}句:")
            for word in count:
                print(word,' ',self.count_tf(word,count)*word_idf_dict[word],'\t')
        print('耗时：%d'%(time.time()-start_time))


import jieba
from jieba.posseg      import cut
test_sentences=['床前明月光',
                '月是故乡明',
                '长空雁叫霜晨月'
                ]  #数据量大的时候才能看不出区别
cut_sentence=[jieba.cut(_,cut_all=False) for _ in test_sentences]
m=myTF_IDF(cut_sentence)
m.print_tfidf_score()
m.more_efficent_print()







