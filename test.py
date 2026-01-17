import torch
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"可用GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"当前GPU名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("PyTorch无法调用GPU")