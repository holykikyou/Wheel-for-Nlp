import numpy as np

class HMM:
    """A、B"""
    def __init__(self,hidden_states,output_states):
        self.M=len(hidden_states)
        self.N=len(output_states)
        self.hidden_states=hidden_states

        self.A=np.zeros((self.M,self.M))#状态转移矩阵

        self.B=np.zeros((self.M,self.N))#发射矩阵
        self.alpha=np.zeros(1,self.M)  #各个隐状态的初始概率






        pass
    def forward_prob(self,out_squence):
        """前向算法"""
        n=len(out_squence)

        a0=out_squence[0]
        """初始概率1*seqlen,分别是"""
        init_prob=self.alpha*np.transpose(self.B[:,a0])   #统计学习方法公式10.15
        prob=init_prob
        for i in range(1,n): #迭代递推的过程
            prob_i=np.zeros_like(prob)
            for cur_state in self.hidden_states:
                for before_state in self.hidden_states:
                    prob_i[cur_state] += prob[before_state] * self.A[before_state][cur_state]
                prob_i[cur_state]=prob_i[cur_state]*self.B[cur_state][out_squence[i]]
            prob=prob_i
        return prob.sum()


    def train(self):
        pass

    def predict(self):
        pass


print(np.random.randint(0,5,3))