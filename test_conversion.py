import numpy as np
import torch

def main():
    ls = np.random.rand(10, 10)
    print("Original array type:", type(ls))
    
    # Force conversion via np.array (with copy=True)
    ls_copy = np.array(ls, copy=True)
    print("After np.array, type:", type(ls_copy))
    
    # Alternatively, try using np.asarray (which should return the same type if ls is already a ndarray)
    ls_asarray = np.asarray(ls)
    print("After np.asarray, type:", type(ls_asarray))
    
    try:
        # Try converting both variants:
        tensor_copy = torch.from_numpy(ls_copy)
        tensor_asarray = torch.from_numpy(ls_asarray)
        print("Conversion succeeded:")
        print("tensor_copy:", tensor_copy)
        print("tensor_asarray:", tensor_asarray)
    except Exception as e:
        print("Error converting array:", e)

if __name__ == "__main__":
    main()
