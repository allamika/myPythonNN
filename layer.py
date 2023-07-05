from typing import Any
from neuron import Neuron

class Layer:
    def __init__(self, insize, outsize) -> None:
        self.neurons = [Neuron(insize) for _ in range(outsize)] 
        
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    

if __name__=="__main__":
    x = [2.0,3.0]
    l = Layer(2,3)
    print(l(x))
    print(len(l.parameters())) # 2*3+3 = 9
    