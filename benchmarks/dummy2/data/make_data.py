import numpy as np

def generate(x, a=1.0, b=-1.0, noise=0.3):
    return a*x + b + noise*np.random.uniform(-0.5,0.5,size=x.shape)

x = np.random.uniform(-4, 4, size=20)
y = generate(x)

for pt in zip(x,y):
    print(pt[0], pt[1])
