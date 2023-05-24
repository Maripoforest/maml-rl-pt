import torch

# Create the input tensor
input_tensor = torch.randn(200, 20, 44)

# Reshape the tensor
reshaped_tensor = input_tensor.view(-1, 44)

# Print the reshaped tensor size
print(reshaped_tensor.size())
