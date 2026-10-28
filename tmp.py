import numpy as np

x = np.arange(1,10).reshape((3,3))

y = np.arange(10,19).reshape((3,3))


print(x)
print("*")
print(y)

res_1 = x @ y

res_2 = np.matmul(x,y)
print()

print(res_2 == res_1)

print(x.size)
print(x.shape)
print(x.dtype)

a = np.array([0, 1, 2],dtype=np.int64)
b = np.array([0, 1, 2],dtype=np.int32)

c = a + b
print(c, c.shape, c.dtype)

print()
print()
print(c)
print(c.shape)
print()

transpose = c.reshape(-1, 1)
print(transpose)
print(transpose.shape)

print()
print()

atranspose = c[:, None]
print(atranspose)
print(atranspose.shape)

