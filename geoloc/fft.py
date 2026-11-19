"""2D real FFT for the coarse search: cuFFT when it works, our own matrix-multiply FFT when it does not.

rfft2(x, size)    FFT of x (zero padded to `size`), keeps the half spectrum, same as torch.fft.rfft2
irfft2(x, size)   inverse of that, same as torch.fft.irfft2

Why this file exists: on some Jetsons every cuFFT call fails (CUFFT_INVALID_SIZE or CUFFT_INTERNAL_ERROR, even for a 64x64 array),
while matrix multiplication works. The fallback is the classic split n = n1 * n2 (Cooley-Tukey). Each step is a matrix multiply with a
small DFT matrix, so it only needs cuBLAS. Setting FFT_BACKEND = "auto" tries cuFFT once and falls back by itself.
"""
import math
import sys

import torch
import torch.nn.functional as functional

import settings

SMALL_LENGTH = 32          # a length up to this is done with one dense DFT matrix
DENSE_INPUT = 256          # an input shorter than this (zero padded) is done with one dense matrix
CHUNK_ELEMENTS = 12000000  # about this many numbers go through the FFT at once (limits the memory)

matrices = {}
backend_choice = {}


# ---------------------------------------------------------------------------
# which FFT to use
# ---------------------------------------------------------------------------

def cufft_works(device):
    try:
        test = torch.zeros(1, 16, 16, device=device)
        torch.fft.rfft2(test)
        if device != "cpu":
            torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def use_matmul_fft(device):
    """True when the matrix-multiply FFT has to be used. Decided once per device, and said once on stderr."""
    wanted = settings.FFT_BACKEND
    if wanted == "matmul":
        return True
    if wanted == "torch":
        return False
    if device not in backend_choice:
        backend_choice[device] = not cufft_works(device)
        if backend_choice[device]:
            print("cuFFT does not work on this machine, using the matrix-multiply FFT", file=sys.stderr, flush=True)
    return backend_choice[device]


def rfft2(x, size):
    if use_matmul_fft(str(x.device)):
        return matmul_rfft2(x, size)
    return torch.fft.rfft2(x, s=size)


def irfft2(x, size):
    if use_matmul_fft(str(x.device)):
        return matmul_irfft2(x, size)
    return torch.fft.irfft2(x, s=size)


# ---------------------------------------------------------------------------
# DFT matrices
# ---------------------------------------------------------------------------

def dft_matrix(rows, cols, length, sign, device):
    """M[j, k] = exp(sign * 2 pi i * j * k / length), for j < rows and k < cols."""
    key = ("dft", rows, cols, length, sign, str(device))
    if key not in matrices:
        if len(matrices) > 200:
            matrices.clear()
        j = torch.arange(rows, dtype=torch.float64)[:, None]
        k = torch.arange(cols, dtype=torch.float64)[None, :]
        angle = sign * 2 * math.pi * ((j * k) % length) / length
        matrix = torch.polar(torch.ones_like(angle), angle)
        matrices[key] = matrix.to(torch.complex64).to(device)
    return matrices[key]


def twiddle(n1, n2, sign, device):
    """T[k1, j2] = exp(sign * 2 pi i * k1 * j2 / (n1 * n2))."""
    key = ("twiddle", n1, n2, sign, str(device))
    if key not in matrices:
        if len(matrices) > 200:
            matrices.clear()
        k1 = torch.arange(n1, dtype=torch.float64)[:, None]
        j2 = torch.arange(n2, dtype=torch.float64)[None, :]
        angle = sign * 2 * math.pi * ((k1 * j2) % (n1 * n2)) / (n1 * n2)
        matrix = torch.polar(torch.ones_like(angle), angle)
        matrices[key] = matrix.to(torch.complex64).to(device)
    return matrices[key]


def pick_factor(length):
    """The biggest divisor of length that is at most SMALL_LENGTH, or 0 when there is none."""
    for factor in range(SMALL_LENGTH, 1, -1):
        if length % factor == 0:
            return factor
    return 0


# ---------------------------------------------------------------------------
# complex FFT along the last axis
# ---------------------------------------------------------------------------

def fft_last_axis(x, length, sign):
    """Complex DFT (not normalised) along the last axis, whose size is `length`."""
    factor = pick_factor(length)
    if length <= SMALL_LENGTH or factor == 0:
        return x @ dft_matrix(length, length, length, sign, x.device)

    n1 = factor
    n2 = length // n1
    leading = x.shape[:-1]
    # index j = j1 * n2 + j2. Step 1: DFT over j1 (a matrix multiply). Step 2: twiddle factors. Step 3: DFT over j2 (the same function again).
    y = x.reshape(*leading, n1, n2)
    y = dft_matrix(n1, n1, n1, sign, x.device) @ y
    y = y * twiddle(n1, n2, sign, x.device)
    y = fft_last_axis(y, n2, sign)
    # the result bin is k = k1 + n1 * k2, so k2 has to come first before the two axes are joined
    y = y.transpose(-1, -2)
    return y.reshape(*leading, length)


def dft_last_axis(x, length, sign, keep):
    """DFT of length `length` along the last axis of x, where x holds only the first x.shape[-1] values (the rest are zeros).
    Returns the first `keep` bins."""
    given = x.shape[-1]
    if given <= DENSE_INPUT or length <= SMALL_LENGTH:
        return x @ dft_matrix(given, keep, length, sign, x.device)
    if given < length:
        x = functional.pad(x, (0, length - given))
    result = fft_last_axis(x, length, sign)
    return result[..., :keep]


# ---------------------------------------------------------------------------
# 2D real FFT and its inverse
# ---------------------------------------------------------------------------

def matmul_rfft2(x, size):
    rows, cols = size
    half = cols // 2 + 1
    leading = x.shape[:-2]
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    out = torch.empty(flat.shape[0], rows, half, dtype=torch.complex64, device=x.device)

    step = max(1, CHUNK_ELEMENTS // (rows * cols))
    for start in range(0, flat.shape[0], step):
        part = flat[start:start + step].to(torch.complex64)
        part = dft_last_axis(part, cols, -1, half)                    # along the columns axis, only the half spectrum
        part = part.transpose(-1, -2)
        part = dft_last_axis(part, rows, -1, rows)                    # along the rows axis
        out[start:start + step] = part.transpose(-1, -2)
    return out.reshape(*leading, rows, half)


def matmul_irfft2(x, size):
    rows, cols = size
    half = cols // 2 + 1
    leading = x.shape[:-2]
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    out = torch.empty(flat.shape[0], rows, cols, dtype=torch.float32, device=x.device)

    step = max(1, CHUNK_ELEMENTS // (rows * cols))
    for start in range(0, flat.shape[0], step):
        part = flat[start:start + step]
        part = part.transpose(-1, -2)
        part = dft_last_axis(part, rows, 1, rows)                     # inverse along the rows axis
        part = part.transpose(-1, -2)
        # a real signal has a mirrored spectrum: bin cols-k is the conjugate of bin k. Rebuild the missing half.
        mirrored = part[..., 1:cols - cols // 2].flip(-1).conj()
        full = torch.cat([part, mirrored], dim=-1)
        signal = fft_last_axis(full, cols, 1)
        out[start:start + step] = signal.real / (rows * cols)
    return out.reshape(*leading, rows, cols)
