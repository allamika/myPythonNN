from value import Value
import numpy as np


class Tensor:
    def __init__(self, data: np.array, operator="", parents=[], label="") -> None:
        self.data = data
        self.grad = np.zeros(data.shape)
        self.shape = data.shape
        self._backward = lambda: None
        self.parents = parents
        self.operator = operator
        self.label = label
    
    def __repr__(self) -> str:
        return f"Tensor(data={self.data:.2f}, gradient={self.grad:.2f})"
    
    def __add__(self, other):
        out = Tensor(self.data+other.data, "add", [self, other])
        
        return out
    
    def __mul__(self, other):
        out = Tensor(self.data@other.data, "mul", [self, other])
        
        return out
    
    def tanh(self):
        x = self.data
        t = (np.exp(2*x)-1) / (np.exp(2*x)+1)
        out = Tensor(t, "tanh", parents=[self])
        
        return out
    
    def sum(self, axis:tuple):
        s = np.asarray(self.data.sum(axis=axis))
        return Tensor(s, f"sum{axis}", parents=self)
    
    
if __name__=="__main__":
    w = Tensor(np.asarray([-3.0,1.0])); w.label="w"
    x = Tensor(np.asarray([2.0,0.0])); x.label="x"
    xmw =  x*w; xmw.label = xmw
    xw = xmw.sum((0))
    b = Tensor(np.asarray(6.8813735870195432)); b.label="b"
    n = xw+b; n.label=n
    o = n.tanh()
    print(o)
    