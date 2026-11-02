import numpy as np


base = np.arange(12).reshape(3, 4)
left = base[:, ::2]
right = base[:, [0, 2]]
ff = base[:, 0:1]
aff = base[:, 0]
aaff = base[:, [0]]

print(np.shares_memory(base, left))
print("--------")
print()
print(np.shares_memory(base, right))
print("--------")
print(np.shares_memory(base, ff))
print("--------")
print(np.shares_memory(base, aff))
print("--------")
print(np.shares_memory(base, aaff))
