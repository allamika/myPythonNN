from layer import Layer

class MLP:
    def __init__(self, ins, outss) -> None:
        sizes = [ins] +  outss
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(outss))]
        
    def __call__(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

if __name__=="__main__":
    x = [2.0,3.0]
    mlp = MLP(2, [4,5,3])
    print(mlp(x))
    print(len(mlp.parameters())) # 2*4+4  +  4*5+5  +  5*3+3  =  55