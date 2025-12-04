import os
import sys

# Add CUDA DLL paths to PATH before importing torch
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if os.path.exists(cuda_path):
    cuda_bin = os.path.join(cuda_path, "bin")
    cuda_lib = os.path.join(cuda_path, "lib", "x64")
    if cuda_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{cuda_bin};{cuda_lib};{os.environ.get('PATH', '')}"
    os.environ["CUDA_PATH"] = cuda_path
    os.environ["CUDA_HOME"] = cuda_path
    print(f"[INFO] CUDA paths added to environment: {cuda_bin}, {cuda_lib}")

try:
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
except OSError as e:
    print(f"[ERROR] Failed to load PyTorch: {e}")
    print("\n[SOLUTION] Try the following:")
    print("1. Install Visual C++ Redistributable 2015-2022:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("2. Restart your terminal/command prompt")
    print("3. Ensure CUDA 12.8 is properly installed")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
    sys.exit(1)

