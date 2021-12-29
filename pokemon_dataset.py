import torch
x = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(":::",x.resize_(2, 2))
print("::::",x.resize_(3, 3))