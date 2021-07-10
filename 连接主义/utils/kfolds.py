"""
5-10
K折交叉验证模板
在鸢尾花数据集使用支持向量机进行试验

以及基于sklearn和
"""
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import numpy as np
np.random.seed(2021-5-10)
iris=load_iris()
X_train,X_test,Y_train,Y_test=train_test_split(iris.data,iris.target,random_state=1)

"""现成工具"""

from sklearn.model_selection import cross_val_score
# print(cross_val_score(clf,X_train,Y_train).mean())
"""手动实现K折交叉验证"""

num=5
kf=StratifiedKFold(n_splits=3,random_state=True,shuffle=False)
"""保存交叉验证结果"""
train_preds=np.zeros(len(X_train))
"""保存k次预测结果并求平均"""
test_preds=np.zeros((len(X_test),num))
# for train,eval in kf.split(X_train):
#     print(len(train))
clf=SVC(C=1,gamma=0.123,kernel='rbf')

"""将训练集切分成五折"""
def kfolds(clf=clf,X_train=X_train,Y_train=Y_train,X_test=X_test):
    for i, (train_index, eval_index) in enumerate(kf.split(X_train, Y_train)):  # kfolds没有序号
        """注意版本问题valueError: Expected 2D array, got 1D array instead:"""
        clf.fit(X_train[train_index], Y_train[train_index])
        """其实是每次验证都填充1/K的预测结果"""
        train_preds[eval_index] = clf.predict(X_train[eval_index])
        test_preds[:, i] = clf.predict(X_test)
    print(f"{clf.__module__}fold,eval_accuarcy:   ", accuracy_score(Y_train, train_preds))
    print(test_preds.mean(axis=1))  # 这个平均值意义不大
    print(test_preds)
    """错误的写法，分类模型"""
    # print(f"{clf.__module__}fold,test_eval_accuarcy:   ", accuracy_score(Y_test, test_preds.mean(axis=1)))
    """查看第二折在测试集的准确率，不过交叉验证本意就是要使用所有数据，所以一般不提前切分测试集，"""
    print(f"{clf.__module__}fold,test_eval_accuarcy:   ", accuracy_score(Y_test, test_preds[:, 1]))




"""学习曲线和验证曲线"""
import matplotlib.pyplot as plt
def draw_curve(params,train_scores,test_scores):
    """根据learning——curve函数输出的分数绘制学习曲线"""
    train_mean=train_scores.mean(axis=1)
    test_mean=test_scores.mean(axis=1)
    train_std=train_scores.std(axis=1)
    test_std=test_scores.std(axis=1)
    print('train_std:',train_std)
    ax1=plt.subplots(1,1)
    print('train_mean:',train_mean)
    try:
        print('ax1:',ax1.index)
    except Exception as e:
        print(e)
    plt.plot(params,train_mean,'--',color='g',label='training')
    plt.plot(params,test_mean,'o-',color='r',label='testing')
    plt.fill_between(params,train_mean+train_std,train_mean-train_std)
    plt.fill_between(params, test_mean + test_std, test_mean - test_std)
    plt.grid()
    plt.legend()
    plt.ylim(0.5,1.05)
    plt.show()




from sklearn.model_selection import learning_curve
""""""

params=np.linspace(0.1,1,10)
"""输出第一维为均匀切分层数，第二维是折数"""
train_size,train_score,test_score=learning_curve(clf,X_train,Y_train,cv=5,train_sizes=params, scoring='accuracy')
# print('train_score:',train_score.shape,'\n','test:',test_score)

draw_curve(params,train_scores=train_score,test_scores=test_score)













