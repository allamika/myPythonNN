from value import Value
import random
import numpy as np



class Neuron:
    
    def __init__(self, ninputs) -> None:
        self.w = [Value(random.uniform(-1,1), label=f"w{i}") for i in range(ninputs)]
        self.b = Value(random.uniform(-1,1))
        
    def __call__(self,x):
        act = sum([wi*xi for wi,xi in zip(self.w, x)], self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]
    

if __name__=="__main__":
    x = [2.0,3.0]
    n = Neuron(2)
    print(n(x))
    print(len(n.parameters())) # 2+1 = 3
    n(x).backward()
    print(n.parameters())