"""LSP smoke: jump to tl.program_id / tl.load should resolve under third_party/triton."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = out.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    # Intentionally not runnable on macOS without a real Triton build.
    print("smoke file for Pylance / go-to-definition only")
