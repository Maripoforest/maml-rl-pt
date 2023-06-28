import torch
tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])

dot_product = torch.dot(tensor1, tensor2)
print("Dot product:", dot_product)
element_wise_product = torch.multiply(tensor1, tensor2)
# or element_wise_product = tensor1 * tensor2
print("Element-wise product:", element_wise_product) 