import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else "N/A"}')
print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f'GPU {i}: {name} (Compute Capability: {capability[0]}.{capability[1]})')

