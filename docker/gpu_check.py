"""GPU check for the starter image (PyTorch). Run it on the device, never during a build.

    docker run --rm --runtime nvidia --entrypoint python3 <image> /opt/starter/gpu_check.py

It makes sure that PyTorch sees the GPU, and that real cuDNN, cuBLAS and
plain CUDA kernels run. On some images, `is_available()` alone passed, but
the first convolution then failed with a cuDNN version mismatch. If a
check fails, the exit code is not zero. The base image has its own check
for CUDA, cuDNN, TensorRT and nvcc: /opt/base/first_boot_check.py.
"""
import sys

import torch


def main():
    print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
    if not torch.cuda.is_available():
        print("FAIL: torch.cuda.is_available() is False. PyTorch runs only on the CPU. "
              "Look for a pip install that replaced the GPU wheel. Also make sure that "
              "the container started with --runtime nvidia.")
        return 1
    print("device:", torch.cuda.get_device_name(0))
    with torch.no_grad():
        x = torch.randn(1, 3, 64, 64, device="cuda")
        y = torch.nn.Conv2d(3, 8, 3).cuda()(x)                 # cuDNN
        a = torch.randn(256, 256, device="cuda")
        (a @ a).sum().item()                                   # cuBLAS
        torch.nn.functional.interpolate(x, scale_factor=0.5)   # plain CUDA kernel
    torch.cuda.synchronize()
    if not torch.isfinite(y).all():
        print("FAIL: convolution produced non-finite values")
        return 1
    free, total = torch.cuda.mem_get_info()
    print(f"GPU OK | memory free {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
