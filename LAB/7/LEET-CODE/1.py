import numpy as np

a = np.array([8,1,2,2,3])
res = []

for elem in a:
    res.append(np.count_nonzero(a < elem))
 
print(res)
    