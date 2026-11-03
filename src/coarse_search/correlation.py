import torch

from header import settings


def fast_length(n):
    assert int(n) == n and n >= 1, "n must be a whole number >= 1, got %s" % n
    while True:
        rest = n
        for factor in (2, 3, 5):
            while rest % factor == 0:
                rest = rest // factor
        if rest == 1:
            return n
        n = n + 1


def correlate_everywhere(window, kernel, cells, min_energy=settings.MIN_ENERGY, min_variance=settings.MIN_VARIANCE):
    assert window.ndim == 3 and kernel.ndim == 3, "window and kernel must be (bins, rows, cols)"
    assert window.shape[0] == kernel.shape[0], "window and kernel need the same number of bins"
    assert kernel.shape[1] <= window.shape[1] and kernel.shape[2] <= window.shape[2], "kernel is bigger than the window"
    assert cells.shape == kernel.shape[1:], "cells must be (rows, cols) of the kernel"
    assert window.device == kernel.device == cells.device, "window, kernel and cells must be on the same device"
    assert cells.sum() > 0, "kernel has no valid cells"
    n_bins, height, width = window.shape
    kernel_h, kernel_w = kernel.shape[1], kernel.shape[2]
    size = (fast_length(height), fast_length(width))
    out_h = height - kernel_h + 1
    out_w = width - kernel_w + 1
    n_cells = cells.sum()

    window_fft = torch.fft.rfft2(window, s=size)
    cells_fft = torch.fft.rfft2(cells, s=size)

    kernel_fft = torch.fft.rfft2(kernel, s=size)
    dot = torch.fft.irfft2((window_fft * kernel_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]
    strength_fft = torch.fft.rfft2((window ** 2).sum(dim=0), s=size)
    energy = torch.fft.irfft2(strength_fft * cells_fft.conj(), s=size)[:out_h, :out_w].clamp_min(0)
    cosine = dot / (kernel.pow(2).sum().sqrt() * energy.sqrt() + 1e-6)

    kernel_mean = kernel.sum(dim=(1, 2), keepdim=True) / n_cells
    kernel_zero = (kernel - kernel_mean) * cells
    zero_fft = torch.fft.rfft2(kernel_zero, s=size)
    dot_zero = torch.fft.irfft2((window_fft * zero_fft.conj()).sum(dim=0), s=size)[:out_h, :out_w]

    squared_sums = torch.zeros_like(energy)
    for b in range(n_bins):
        local_sum = torch.fft.irfft2(window_fft[b] * cells_fft.conj(), s=size)[:out_h, :out_w]
        squared_sums += local_sum ** 2 / n_cells
    variance = (energy - squared_sums).clamp_min(0)
    zncc = dot_zero / (kernel_zero.pow(2).sum().sqrt() * variance.sqrt() + 1e-6)

    no_map = energy < min_energy * n_cells
    too_flat = variance < min_variance * n_cells
    cosine[no_map] = -1.0
    zncc[no_map | too_flat] = -1.0
    return {"cosine": cosine, "zncc": zncc}
