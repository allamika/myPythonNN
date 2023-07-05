from mlp import MLP
from gradient_descent import GD
from matplotlib import pyplot as plt
import numpy as np


def gen_data():
    x = np.random.uniform(-1,1,100)
    y = np.random.uniform(-1,1,100)
    z = np.exp(-(x**2 + y**2))
    separation = np.median(z)
    z = np.asarray(list(map(lambda x: 1 if x>separation else -1, z)))
    filtefg1 = z==1
    filtefg2 = z==-1
    grp1x, grp1y = x[filtefg1], y[filtefg1]
    grp2x, grp2y = x[filtefg2], y[filtefg2]
    #plt.scatter(grp1x, grp1y)
    #plt.scatter(grp2x, grp2y)
    #plt.show()
    X = np.concatenate((np.expand_dims(x,1), np.expand_dims(y,1)), axis = 1)
    Y = z
    return X, Y


X, Y = gen_data()

mlp = MLP(2, [10, 4, 1])
optimizer = GD(mlp.parameters(), 0.01)
all_loss = []
bs = 8
for k in range(60):
    eploss = 0
    
    for i in range(int(len(X)/bs)):
        Xb = (X[(i)*bs:(i+1)*bs,:]).tolist()
        Yb = (Y[(i)*bs:(i+1)*bs]).tolist()
        
        ypred = [mlp(x) for x in Xb]
        loss = sum((yout - ygt)**2 for ygt, yout in zip(Yb, ypred)); loss.label = "loss"
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        eploss += loss.data
        
    print(k, eploss)
    all_loss.append(eploss)

plt.plot(all_loss);plt.title("Deep Learning with batch"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.show()

