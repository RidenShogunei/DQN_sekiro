import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available")
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")