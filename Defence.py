import torch

def trimmed(y_values):
    # Get the list of neighbors and their corresponding tensors
    neighbors = list(y_values.keys())
    tensors = [y_values[j] for j in neighbors]  # List of tensors

    # Stack tensors to create a tensor of shape (num_neighbors, ...)
    tensor_stack = torch.stack(tensors, dim=0)  # Shape: (num_neighbors, ...)

    # Compute the maximum and minimum values across neighbors at each tensor position
    max_tensor, _ = torch.max(tensor_stack, dim=0, keepdim=True)  # Shape: (1, ...)
    min_tensor, _ = torch.min(tensor_stack, dim=0, keepdim=True)  # Shape: (1, ...)

    # Create masks for positions where tensors have max or min values
    max_mask = (tensor_stack == max_tensor)
    min_mask = (tensor_stack == min_tensor)

    # Combine masks to identify positions to zero out
    mask = max_mask | min_mask  # Shape: (num_neighbors, ...)

    # Create a copy of the tensor stack to adjust
    adjusted_tensor_stack = tensor_stack.clone()

    # Set max and min values to zero at the corresponding positions
    adjusted_tensor_stack[mask] = 0

    # Convert the adjusted tensor stack back to a list of tensors for each neighbor
    adjusted_tensors = [adjusted_tensor_stack[k] for k in range(adjusted_tensor_stack.shape[0])]

    # Map the adjusted tensors back to their respective neighbors
    adjusted_y_values = {j: adjusted_tensors[k] for k, j in enumerate(neighbors)}

    return adjusted_y_values
