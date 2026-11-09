# Team image of Main_Pipeline_Headless, adapted from the starter kit Dockerfile (torch 2.8.0 + torchvision 0.23.0 Jetson wheels, numpy, OpenCV, plus our requirements.txt).
# GPU support, plus OpenCV, built FROM the base image. The image is only your
# environment. The harness mounts your code (code/, with the entry point
# code/run) and weights (models/) at run time (interface/SPEC.md). You do
# not have to use PyTorch. The base has only the system stack (CUDA,
# cuDNN, TensorRT, Python 3.10). Thus, you can delete step 1 and install
# the packages that you need.
#
# On a Jetson, build it like any image:
#   docker build -t myteam:dev .
# This needs BuildKit, because the RUN steps use --mount=type=cache: Docker
# with the buildx plugin (check with `docker buildx version`). Without it,
# the build stops with "the --mount option requires BuildKit". Docker's
# own packages include the plugin; Ubuntu's docker.io package may not.
# The pinned versions below are the defaults, the same as in
# base/versions.env. On a PC, base/build.sh cross-builds for the Orin and
# builds the x86 version:
#   base/build.sh starter-arm64 --context ../myteam --tag myteam:dev
#   base/build.sh starter-x86 --context ../myteam --tag myteam:x86
#
# Rebuild the image only when your dependencies change. Changes to code
# and weights need no rebuild.
ARG BASE_IMAGE=uav-geoloc-base:jp6-arm64
FROM ${BASE_IMAGE} AS base-amd64
FROM ${BASE_IMAGE} AS base-arm64
# A cross-build has no GPU, so CUDA extensions that you compile during
# the build need the architecture of the Orin, given explicitly. On x86,
# PyTorch finds the architecture of your GPU.
ENV TORCH_CUDA_ARCH_LIST=8.7
FROM base-${TARGETARCH}

ARG TARGETARCH
ARG TORCH_WHL_ARM64="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl#sha256=62a1beee9f2f147076a974d2942c90060c12771c94740830327cae705b2595fc"
ARG TORCHVISION_WHL_ARM64="https://pypi.jetson-ai-lab.io/jp6/cu126/+f/907/c4c1933789645/torchvision-0.23.0-cp310-cp310-linux_aarch64.whl#sha256=907c4c1933789645ebb20dd9181d40f8647978e6bd30086ae7b01febb937d2d1"
ARG CUSPARSELT_URL_ARM64="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz"
ARG CUSPARSELT_SHA256_ARM64="d3b0a660fd552e0bd9a4491b15299d968674833483d5f164cfea35e70646136c"
ARG TORCH_WHL_X86="https://download-r2.pytorch.org/whl/cu126/torch-2.8.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=6bf3c0085af4176137f216c39995dede9beda9af1307fd1dee2305f4f351eb42"
ARG TORCHVISION_WHL_X86="https://download-r2.pytorch.org/whl/cu126/torchvision-0.23.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl#sha256=29750c7b2236e1714823b3441a64c498d84366672c9f09a87d72bd2be8867080"
ARG NUMPY_VERSION="1.26.4"
ARG OPENCV_VERSION="4.10.0.84"

# 1a. System libraries that the PyTorch and torchvision wheels load. The
#     Jetson torch wheel links OpenBLAS, and the image I/O of torchvision
#     uses libjpeg and libpng. The base has none of them.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 libgomp1 libjpeg-turbo8 libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# 1b. PyTorch with GPU support.
#   - arm64: NVIDIA's Jetson wheels (Python 3.10 only). `pip install torch`
#     from pypi.org is CPU-only on arm64. torch >= 2.6 also needs
#     cuSPARSELt, which JetPack 6 does not ship.
#   - amd64: the CUDA 12.6 wheels from download.pytorch.org.
#   pip checks each wheel against the #sha256= in its URL.
RUN --mount=type=cache,target=/root/.cache/pip \
    set -eu; \
    if [ "${TARGETARCH}" = "arm64" ]; then \
        curl -fsSL "${CUSPARSELT_URL_ARM64}" -o /tmp/cusparselt.tar.xz; \
        echo "${CUSPARSELT_SHA256_ARM64}  /tmp/cusparselt.tar.xz" | sha256sum -c -; \
        mkdir -p /tmp/cusparselt; \
        tar -xJf /tmp/cusparselt.tar.xz -C /tmp/cusparselt --strip-components=1; \
        cp -a /tmp/cusparselt/lib/*.so* /usr/local/cuda/lib64/; \
        rm -rf /tmp/cusparselt*; for i in 1 2 3; do timeout 120 ldconfig && break; done; \
        pip3 install "${TORCH_WHL_ARM64}" "${TORCHVISION_WHL_ARM64}" "numpy==${NUMPY_VERSION}"; \
    else \
        pip3 install "${TORCH_WHL_X86}" "${TORCHVISION_WHL_X86}" "numpy==${NUMPY_VERSION}"; \
    fi

# 1c. Freeze torch, torchvision and numpy. PIP_CONSTRAINT applies to every
#     later `pip install`. Thus, if a package needs a different torch or
#     numpy, the build fails and the GPU build does not change.
RUN mkdir -p /opt/starter && python3 -c "import importlib.metadata as m; \
print('\n'.join(f'{d}=={m.version(d)}' for d in ('torch', 'torchvision', 'numpy')))" \
    > /opt/starter/constraints.txt && cat /opt/starter/constraints.txt
ENV PIP_CONSTRAINT=/opt/starter/constraints.txt

# 1d. OpenCV and your other dependencies. Pin exact versions and use
#     only wheels, because a compile under QEMU is very slow.
COPY requirements.txt /opt/starter/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install "opencv-contrib-python-headless==${OPENCV_VERSION}" -r /opt/starter/requirements.txt

# Import-only check, no GPU. Under QEMU this proves the packages resolve.
# On the device, run the GPU check:
#   docker run --rm --runtime nvidia --entrypoint python3 <image> /opt/starter/gpu_check.py
COPY gpu_check.py /opt/starter/gpu_check.py
RUN python3 -c "import torch, torchvision, cv2, numpy; \
print('starter imports OK | torch', torch.__version__, '| torchvision', torchvision.__version__, '| cv2', cv2.__version__, '| numpy', numpy.__version__)"

# Offline at run time: the evaluation runs with --network none. Libraries
# that usually download weights read them from the mounted models/ folder.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/data/models/hf TORCH_HOME=/data/models/torch

# Imports of our own pipeline (kornia LoFTR, timm DINO, rasterio, pyproj). Import only, no GPU needed, so it also runs under QEMU.
RUN python3 -c "import kornia, timm, rasterio, pyproj, einops, safetensors; \
from kornia.feature import LoFTR; \
print('pipeline imports OK | kornia', kornia.__version__, '| timm', timm.__version__, '| rasterio', rasterio.__version__, '| pyproj', pyproj.__version__)"

# No ENTRYPOINT: the evaluation always starts /data/code/run.
