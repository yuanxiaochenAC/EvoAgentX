import numpy as np

def sample_uniform_01(n=10):
    print(np.random.uniform(low=0.0, high=1.0, size=n))

sample_uniform_01()