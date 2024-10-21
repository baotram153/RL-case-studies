import numpy as np

arr = np.array([[1,2,3], [4,5,6]])

arr1 = arr.cumsum(axis=0)
arr2 = arr.flatten('F')

print(arr2)
print(arr1)

# import pandas as pd
# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt

# x = np.array([1, 2, 3])
# y = np.array([[4, 6, 8], [1,2,3], [3,4,5]])

# x = x[None]
# y = y[None]
# mat = np.concatenate([x.T, y.T], axis=1)
# print(mat)

# fig, axis = plt.subplots(1, 1)

# df = pd.DataFrame(mat, columns=["x", "y"])
# print(df)
# sns.lineplot(df, ax=axis, x="x", y="y")

# plt.show()

# import torch
# import numpy as np

# mat1 = torch.zeros((4, 1, 3))
# mat2 = torch.zeros((4, 1, 3, 3))

# mat3 = mat1.matmul(mat2)
# print(mat3.shape)

print(f"zfill test: " + f"{2}".zfill(3))