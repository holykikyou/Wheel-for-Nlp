
import numpy as np
import scipy.sparse as sp
def cosine_similarity(x,y):
    """余弦相似度计算"""
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    print(rowsum)
    r_inv = np.power(rowsum, -1)#.flatten()
    print(r_inv)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv=r_inv
    #r_mat_inv = sp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx



x=np.array(list(range(1,11)),dtype=np.float)
print(np.power(x,-1))
x=np.mat(x)
print(normalize(x))
y=np.random.rand(1,10)
print(x,y,cosine_similarity(x,y))