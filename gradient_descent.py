from value import Value
from mlp import MLP
from matplotlib import pyplot as plt

class GD():
    def __init__(self, parameters, lr) -> None:
        self.params = parameters
        self.lr = lr
        
    def step(self):
        for parm in self.params:
            parm.data += -self.lr * parm.grad
    
    def zero_grad(self):
        for parm in self.params:
            parm.grad = 0.0
            

if __name__=="__main__":
    mlp = MLP(3, [4, 4, 1])
    xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0] # desired targets
    
    
    optimizer = GD(mlp.parameters(), 0.05)
    all_loss = []
    for k in range(20):
        ypred = [mlp(x) for x in xs]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred)); loss.label = "loss"
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()        

        print(k, loss.data)
        all_loss.append(loss.data)
    
    print(ys, list(map(lambda x: x.data,ypred)))
    plt.plot(all_loss);plt.title("My first Deep Learning"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()
    