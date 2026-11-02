import numpy as np

import numpy as np

x = np.arange(3500).reshape(250,14)

print(x[1:, :].shape)
print()
print(x[:, 1].shape)
print()
print()
print(x[[-1], :].shape)
print()