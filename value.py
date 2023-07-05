import math



class Value:
    def __init__ (self, data:float, operator='', parents=[], label=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.parents = parents
        self.label = label
        self.operator = operator
        
    def __repr__(self) -> str:
        return f"Value(data={self.data:.2f}, gradient={self.grad:.2f}, label={self.label})"
        
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, "add", parents=[self, other])
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, "mul", parents=[self, other])
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self*-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, f"**{other}", parents=[self])
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out

    def exp(self):
        out = Value(math.exp(self.data),"exp", parents=[self])
        
        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t,"tanh", parents=[self])
        
        def _backward():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        return out
    
    def topo_graph(self, visited=None, topo=None):
        if visited is None: visited = []
        if topo is None: topo = []
        
        if self not in visited: 
            visited.append(self)
            for parent in self.parents:
                parent.topo_graph(visited,topo)
            topo.append(self)
        return topo
    
    def backward(self):
        topo = self.topo_graph([])
        self.grad = 1
        for i in reversed(topo):
            i._backward()
            
        

if __name__ == "__main__":
    x1 = Value(2.0, label='x1')
    w1 = Value(-3.0, label='w1')
    x2 = Value(0.0, label='x2')
    w2 = Value(1.0, label='w2')
    x1w1 = x1 * w1; x1w1.label = "x1w1"
    x2w2 = x2 * w2; x2w2.label = "x2w2"
    b = Value(6.8813735870195432, label="b")
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
    n = x1w1x2w2 + b; n.label = "n"
    o = n.tanh(); o.label = "o"

    back = o.backward()
    print("back prop sur o\n", o.topo_graph(), '\n')
    
    
    #x1 = Value(2.0, label='x1')
    #w1 = Value(-3.0, label='w1')
    #x2 = Value(0.0, label='x2')
    #w2 = Value(1.0, label='w2')
    #x1w1 = x1 * w1; x1w1.label = "x1w1"
    #x2w2 = x2 * w2; x2w2.label = "x2w2"
    #b = Value(6.8813735870195432, label="b")
    #x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = "x1w1x2w2"
    #n = x1w1x2w2 + b; n.label = "n"
    #e = (2*n).exp()
    #o = (e-1)/(e+1); o.label 1 - t**2= "o"

    #back = o.backward()
    #print("back prop sur o\n", o.produce_topo_graph())
    