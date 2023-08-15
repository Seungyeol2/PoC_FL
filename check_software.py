import torch
import sys
# PyTorch 버전 확인
print("PyTorch version:", torch.__version__)

# CUDA 사용 가능 여부 확인
if torch.cuda.is_available():
    # CUDA 버전 확인
    print("Python version:", sys.version)
    # CUDA 버전 확인
    print("CUDA version:", torch.version.cuda)
    # 사용 가능한 GPU 개수 확인
    print("Number of GPUs:", torch.cuda.device_count())
    # 현재 GPU의 이름 확인
    print("Current GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")