import pickle

# Path to the .pkl file
file_path = '/Users/tcpba/MolTransformer_repo/MolTransformer/model/models/index_path/char2ind.pkl'

# Open and load the .pkl file
with open(file_path, 'rb') as file:
    char2ind = pickle.load(file)

# Now, you can use the `char2ind` dictionary
print(char2ind)

print(len(char2ind))


x = 4*30*30 + 2*30*100 + 6*30 +100 
print(x)