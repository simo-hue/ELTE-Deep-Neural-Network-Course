import numpy as np

x = np.arange(50).reshape(5, 10)
y = np.arange(50).reshape(2, 5, 5)

print("x:")
print(x)
print()

print("y:")
print(y)
print()


print((x[:, :, None, None, None] + y))
print()

