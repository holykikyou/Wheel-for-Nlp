"""
reference:HCT_NLPBOOK
构建skip_gram模型
date: 2021-07-01
理解经典的抽样思想和建模思路

最终的目的就是学习隐层的权重矩阵，通过权重矩阵与单热向量相乘就可以得到每个单词的降维后的稠密向量，这个就是词向量。
这个把高维稀疏向量映射为低维稠密向量的过程就叫做嵌入，也就是词嵌入
"""

import torch
from torch import nn

import torch.nn.functional as F
import numpy as np

def create_sample_table():
    """
    构建负采样表
    不同于原本每个训练样本更新所有的权重，负采样每次让一个训练样本仅仅更新一小部分的权重，这样就会降低梯度下降过程中的计算量。
    """
    pass


class Skip_Gram(nn.Module):
    def __init__(self,vocab_size,embed_size,word_count):
        super(Skip_Gram,self).__init__()


    def forward(self,centers,contexts):
        """输入中心和上下文（独热）向量batch"""

        scores=torch.bmm()
