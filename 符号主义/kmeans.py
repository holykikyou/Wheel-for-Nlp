"""2021-06-14
最经典的聚类算法

1、随机产生聚类初始点
2、对各个点计算和各个聚类中心的距离，取最小值对应的中心，保存这个点对应的聚类中心，
3、对有相同中心的点簇重新计算中心点，并作为下一个迭代的聚类中心
"""

import numpy as np
from tqdm import tqdm
np.random.seed(1)

def cluster(x,k,epoch_num):
    cluster_centers=[]   #返回最后迭代完的聚类中心列表
    print("shape:",x.shape)# num,dimension
    indices=np.random.randint(0,x.shape[0],k)    #表示聚类中心的下标,
    assert len(indices)==k   #必须保证随机获得k个不一样的聚类中心，不能重复06-14，一个非常隐秘的bug
    print('indices:',indices)
    for i in indices:
         cluster_centers.append(x[i,:])   #也可以不初始化，每次清零即可

    newclasses=[]  #指的是每个数据对应的类别,长度是shape【0】
    for e in tqdm(range(epoch_num)):
        #cluster_centers.clear()
        newclasses.clear()
        for i in range(x.shape[0]):
            dist = []  # 指每个数据到每个当前聚类点的距离，长度是聚类中心数量,
            for j in range(k):
                dist.append(np.linalg.norm(x[i, :]-cluster_centers[j])) #注意计算公式,
                #列表不能写成[j,:]形式

            newclasses.append(np.argsort(dist)[0])    #将目前距离最小的聚类中心视为该点的中心
            # 之后就是找出其中等于某类别的点，然后重新计算
            # 获取新聚类点

        for j in range(k):
            xj = [a for a, b in enumerate(newclasses) if b == j]  # 获取第j类聚类中向量的序号，注意这个序号和最开始输入是对齐的
            # a是要获取的序号，b对应的类别
            print(f'第{e}次迭代第{j}个聚类列表:{xj}')
            cluster_centers[j] = np.mean(x[xj, :], 0)  #也可以不初始化，像下面那样每次清零,这是默认可以把最前面k个数当做是聚类中心
            #cluster_centers.append(np.mean(x[xj, :], 0))


    print('聚类结果：',cluster_centers)
    print("原始数据对应的类别：",newclasses)
    return cluster_centers,newclasses

def get_center(group_k):
    """中心评估函数,获取一个簇的中心点"""
    pass

def predict(x_test,cluster_centers):
    """使用获得的cluster_centers即聚类中心结果计算和测试数据的距离"""
    pass




x=np.random.randint(0,1000,99).reshape(33,3)  #一共33个三维数据
print(x)
centers,newclasses=cluster(x,4,2)

"""
聚类结果： [array([293.66666667, 189.33333333, 523.33333333]), array([266.42857143, 727.42857143, 709.        ]), array([673.27272727, 726.54545455, 499.        ]), array([558.33333333, 333.55555556, 203.44444444])]
原始数据对应的类别： [0, 1, 2, 3, 2, 3, 0, 1, 3, 2, 3, 2, 2, 3, 1, 1, 3, 2, 2, 3, 2, 1, 3, 0, 1, 3, 0, 2, 2, 0, 2, 0, 1]
"""











