"""
本质上是一个矩阵运算问题

matrix.I要先用np.mat转换
"""

import numpy  as np
import operator
from os import listdir

def optimize(x,y):
    """输入x二维矩阵，和y向量"""
    x=convert_mat_with_x0(x)
    print(x.shape)
    xt=np.transpose(x)
    print(xt.shape)
    xtx=xt.dot(x)
    print('xtx:',xtx)
    inv_XtX=np.linalg.inv(xtx)
    param=inv_XtX.dot(np.transpose(x)).dot(y)
    # #param=np.transpose(x).dot(x).I.dot(np.transpose(x)).dot(y)报错，因为只有先把它转换为mat对象才有I
    # #AttributeError: 'numpy.ndarray' object has no attribute 'I'
    return param


def caculate_value(x_test,param):
    """ 预测函数 """
    x=np.c_[np.ones([x_test.shape[0]]),x_test]  #代表加上偏置项
    y=np.dot(x,param)
    return y


def convert_mat_with_x0(x):
    x=np.c_[np.ones([x.shape[0]]),x]

    print('增加偏置项后形状：',x.shape)
    return x


#x=np.arange(1,21,1).reshape(10,2)  #会报错
# numpy.linalg.LinAlgError: singular matrix
# 两维数据会导致中间运算结果为奇异矩阵，要打乱数据才行
#x=np.random.randint(0,1000,20).reshape(10,2)  #打散数据就完美运行
# x=np.random.randint(0,20,20).reshape(10,2)  #但一般来说这里面不能有重复的数，用choice
x=np.random.choice(20,20,replace=False).reshape(10,2)  #choice也会有重复,所以要修改默认参数 2021-06-14
# x=np.arange(1,11,1).reshape(10,1)
print('原始x:',x)
print('增加偏置项后',np.c_[np.ones([10]),x])
# x=convert_mat_with_x0(x)

# xtx=np.transpose(x).dot(x)  #该结果不能等于奇异矩阵

# print('xtxshape',xtx.shape,xtx)

# x=np.mat([[1,2,5],[2,2,7],[3,4,8]])
# print('x_inv:',np.linalg.inv(np.mat(xtx)))   #只能用于方方阵求逆？这个
y=np.mat(np.random.rand(10,1))


p=optimize(x,y)
print("参数为",p)



def calc_rmse(y_pred,y):
    """评估函数
    计算预测值和真实值得误差"""
    n=y.shape[0]
    return np.sqrt(np.sum((y_pred-y)**2)/n)

def np_c__Test():
    """
    测试拼接数组的函数
    """
    a = np.arange(10).reshape(2, 5)
    print(np.ones([2]))
    b = np.c_[np.ones([2,4]), a]  # 直接拼接两个数组
    print(b)
    """
    [[1. 1. 0. 1. 2. 3. 4.]
     [1. 1. 5. 6. 7. 8. 9.]]
    """
# np_c__Test()
