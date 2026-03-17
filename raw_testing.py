import scipy.io as sio
import numpy as np

# load mat file
data = sio.loadmat('data_food101_demo/ingredient_all_feature.mat')

print("Keys in MAT file:")
print(data.keys())

mat = data['ingredient_all_feature']

print("\nMatrix shape:")
print(mat.shape)

print("\nUnique values:")
print(np.unique(mat))

print("\nFirst row example:")
print(mat[0])