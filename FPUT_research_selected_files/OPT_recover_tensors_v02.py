import numpy as np
import os

def load_tensor(name, directory):
    """
    Load a tensor from its sparse representation.
    
    Args:
        name: Name of the tensor to load
    
    Returns:
        The reconstructed full tensor
    """
    # Load the sparse representation
    data = np.load(os.path.join(directory, f"{name}.npz"))
    
    # Reconstruct the full tensor
    tensor = np.zeros(data['shape'])*1j
    indices = tuple(data['indices'])
    tensor[indices] = data['values']
    
    return tensor

def load_all_tensors(N, alpha, beta):
    """
    Load all tensors from the tensors directory.
    
    Returns:
        Dictionary mapping tensor names to loaded tensors
    """
    
    # Variables to export to main code
    global N_stored, alpha_stored, beta_stored
    
    N_stored = N
    alpha_stored = alpha
    beta_stored = beta
    
    print("Loading tensors from sparse format...")
    tensors = {}
    
    directory = f"tensors_fermi/tensors_opt_N={N}_alpha={alpha}_beta={beta}"

    
    # Ensure directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} not found")
    
    # Load each tensor
    for filename in os.listdir(directory):
        if filename.endswith(".npz"):
            tensor_name = filename[:-4]  # Remove .npz extension
            tensors[tensor_name] = load_tensor(tensor_name, directory)
            
            # Create global variable
            globals()[tensor_name] = tensors[tensor_name]
            
            print(f"Tensor {tensor_name} loaded with shape {tensors[tensor_name].shape}")
            
    print("All tensors loaded successfully!")
    
    
    return tensors


# MAIN

# Example usage
#load_all_tensors(9, 1, 0.05)


