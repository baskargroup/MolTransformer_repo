import numpy as np
import torch

def main():
    # Create a random NumPy array
    ls = np.random.rand(10, 10)
    print("Original array flags:")
    print(ls.flags)

    # If the array is not C-contiguous, make it contiguous
    if not ls.flags['C_CONTIGUOUS']:
        ls = np.ascontiguousarray(ls)
        print("Converted to C-contiguous array.")

    # Try converting to a PyTorch tensor
    try:
        torch_tensor = torch.from_numpy(ls)
        print("Successfully converted NumPy array to torch tensor:")
        print(torch_tensor)
    except Exception as e:
        print("Error converting array:", e)

if __name__ == "__main__":
    main()

