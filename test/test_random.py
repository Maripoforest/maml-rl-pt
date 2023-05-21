import numpy as np
from tqdm import trange
np.random.seed(42)
array = 2 * np.random.binomial(1, p=0.5, size=(1500, 20)) - 1
for arr in trange(40):
    print(array[arr])