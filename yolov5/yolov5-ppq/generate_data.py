import numpy as np

for i in range(512):
    np.save(file = f'./working/data/{i+1}',
            arr=np.random.random(size=[1, 3, 640, 640]))