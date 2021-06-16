import numpy as np
class Hmm():
    def __init__(self,hidden_states,output_state):
        self.hidden_states=hidden_states
        self.output_state=output_state
        self.tran_matrix=np.zeros(self.hidden_states,self.hidden_states)
        self.output_matrix=np.zeros(self.hidden_states,self.output_state)

    def forward_prob(self,outsequence):
        """实现前向算法"""

import numpy as np

print(np.random.randint(0,5,3))