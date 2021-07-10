
"""
2021-06-14
高斯混合模型
将复杂问题简化，分段解决，弄清逻辑和定义

步骤
1、随机分配数据
2、计算k个高斯分布参数，包括协方差矩阵和特征均值
3、计算数据属于某类的概率

新工具函数：
np.loadtxt()类似于pd.readcsv，不过可以直接加载为numpy对象

"""
import numpy as np
from tqdm import tqdm
np.random.seed(1)



def gaussian_fun(x:np.array,u:np.array,sigma):
   """计算高斯函数"""
   d=len(x)#或者x.shape[0]
   left_term=1/(np.power(np.pi,d/2)*np.power(np.linalg.det(sigma),0.5))  #h需要计算协方差矩阵的行列式
   right_term=np.exp(-0.5*np.dot(np.transpose(x-u),np.linalg.inv(sigma),x-u))  #点积
   return left_term*right_term


def gmmcluster(x,k,epoch_num):
    random_labels=np.random.randint(0,k,x.shape[0])   #这里是取0到k之间值随机生成x.shape[0]个数的列表，和kmeans不完全一样
    assert set(random_labels)==k
    """  发现了一个致命的bug,即未必能保证一定能遍历到k个不同的点，尤其是在k和数据量比较接近的情况下"""
    u=[] #不同类的特征均值
    sigma=[] #不同类的协方差矩阵
    for i in range(k):
        xi=x[random_labels==i,:]   #获取等于这个随机标签的下标
        ui=np.mean(xi,0)#计算不同特征的均值，自然是按列计算
        sigma_i=np.cov(np.transpose(xi))  #计算第i类聚类的协方差矩阵，转置后每行都是一个Rir
        u.append(ui)
        sigma.append(sigma_i) #保存到各类的特征均值和协方差矩阵列表

    for e in range(epoch_num):
        new_u=[]
        new_sigma=[]
        for i in range(k):
            ni=0 #归一化
            new_ui=0
            new_sigma_i=np.zeros([x.shape[1],x.shape[1]])   #初始化新协方差矩阵
            for j in range(x.shape[0]):
                pj = 0
                for s in range(k):  # 第三重循环是为了计算概率和
                    pj += gaussian_fun(x[j], u[s], sigma[s])  # 属于各类的概率和
                pij = gaussian_fun(x[j], u[i], sigma[i]) / pj   #数据j属于i类的概率
                ni+=pij #
                new_ui+=pij*x[j]   #相当于求期望，np.mean相当于pij==1/n
                new_sigma_i+=pij*np.array(np.transpose(x[j]-u[i]).dot((x[j]-u[i])))
                #对这个i类依次使用数据计算协方差的期望
                #矩阵加法，要结合公式去计算，非常细节
                new_ui=new_ui/ni
                new_sigma_i=new_sigma_i/ni
                new_u.append(new_ui)
                new_sigma.append(new_sigma_i)
            print(f'第{i}类有{ni}')
        u=new_u
        sigma=new_sigma
    return u,sigma  #返回迭代完成后的特征均值和协方差矩阵列表


def show_cluster_ans(x,k,u,sigma):
    """
    基于协方差矩阵和特征均值矩阵计算聚类结果:
    使用相同的代码计算概率
    取最大概率对应的类型
    最后返回聚类结果列表，

    """
    clusters_ans={}
    for i in range(k):
        clusters_ans[i]=[]

    pass

"""numpy函数测试"""

def np_cov_test():
    a=np.arange(8).reshape(2,4)
    print(a)

    print(np.cov(a))
    print(np.transpose(a))
    print(np.cov(np.transpose(a)))


# np_cov_test()
def np_mat_dot():
    """
    测试np.mat和dot用于矩阵相乘函数
    二维点积,其实就是二维矩阵乘法
    np.transpose不会增加维数，别搞错了
    """
    a = np.arange(8).reshape(2, 4)
    print(np.transpose(a[0]))

    b=np.arange(8).reshape(4,2)
    print('a', a, '\nb:', b)
    #print('矩阵点积：',a*b)
    print('使用np.mat函数')
    res=np.mat(a).dot(b) #(2,4).dot(4,2)
    print(res)

    """一维向量点积和(1,n)*(n,1)乘法是一样的"""
    print(a[0].dot(np.transpose(a[0])))
    print(a[0].dot(a[0]))
    print(np.transpose(a[0]) .dot(a[0]))
    print(np.expand_dims(a[0],1))
    """
    [[0]
 [1]
 [2]
 [3]]
    """
    print(a[0].shape)  #(4,)
    print(np.transpose(np.expand_dims(a[0], 1)))  #shape:(1,4)
    print(np.transpose(np.expand_dims(a[0],1))*(a[0]))    #(1,4)*(4,)
    #print(np.mat(np.transpose(a[0])) .dot(np.mat(a[0])))  #w维度错误
    """
    a [[0 1 2 3]
 [4 5 6 7]] 
b: [[0 1]
 [2 3]
 [4 5]
 [6 7]]
使用np.mat函数
[[28 34]
 [76 98]]
    """

np_mat_dot()

np























