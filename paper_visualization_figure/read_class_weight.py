import pickle
import numpy as np

# Path to the class_weight file
class_weight_path = 'MolTransformer/model/models/index_path/class_weight'

# Load the class_weight data
with open(class_weight_path, 'rb') as file:
    class_weight = pickle.load(file)

# Display the loaded class_weight
print("class_weight")
print(class_weight[0:10])
print(class_weight[10:20])
print(class_weight[0:10])
print('c',class_weight[18])
print('c_',class_weight[17])

print('h',class_weight[21])
print('h_',class_weight[20])

print('min',min(class_weight))
print('argmin',np.argmin(class_weight))
print('argmax',np.argmax(class_weight))

print('max',np.max(class_weight))

ind2char_path = 'MolTransformer/model/models/index_path/ind2char.pkl'

# Load the class_weight data
with open(ind2char_path, 'rb') as file:
    ind2char_path = pickle.load(file)

# Display the loaded class_weight
print("ind2char_path")
print(ind2char_path)

print(ind2char_path[np.argmin(class_weight)])

print(ind2char_path[np.argmax(class_weight)])




