import torch
print(torch.cuda.is_available())  # Should be True if GPU is recognized
print(torch.cuda.device_count())  # Number of GPUs detected
print(torch.version.cuda)  # CUDA version for PyTorch

