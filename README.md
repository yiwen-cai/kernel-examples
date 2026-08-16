# kernel-examples

在 Mac 上写 Triton / CUDA / CuTe 模板 kernel：跳转、补全能用，**不能编译、不能跑**。

CUDA Toolkit 和 CUTLASS 头文件、Triton 源码都由 [`scripts/bootstrap_lsp.sh`](scripts/bootstrap_lsp.sh) 拉到 `third_party/`（不进 git）。没有 GPU、没有 `nvcc`。

## 快速开始

需要：`uv`、`git`、`rsync`、Python ≥ 3.12、系统 `clangd`。

```bash
./scripts/bootstrap_lsp.sh
```

然后在 Cursor 里选解释器 `.venv/bin/python`，Reload Window。用下面三个 smoke 文件确认 Go to Definition：

| 栈 | 文件 | 应能跳到 |
|---|---|---|
| Triton | [`examples/triton/smoke_add.py`](examples/triton/smoke_add.py) | `tl.load` |
| CUDA | [`examples/cuda/smoke_add.cu`](examples/cuda/smoke_add.cu) | `cudaMalloc` |
| CuTe | [`examples/cute/smoke_tensor.cu`](examples/cute/smoke_tensor.cu) | `cute::make_tensor` |

## 目录

```
kernels/                 正式模板，一个 kernel 一个文件
  cuda/                  raw CUDA（cuda_runtime.h）
  cute/                  CuTe / CUTLASS DSL（cute/tensor.hpp）
  triton/                @triton.jit
examples/                LSP smoke，不要塞正式 kernel
scripts/bootstrap_lsp.sh
third_party/             启动脚本生成，大部分不提交
```

层次是 **栈 → 算子 → 文件**，最多三层。同算子跨语言对照靠相对路径，例如：

- [`kernels/cuda/quant/int8_per_token.cu`](kernels/cuda/quant/int8_per_token.cu)
- [`kernels/triton/quant/int8_per_token.py`](kernels/triton/quant/int8_per_token.py)

CUDA 和 CuTe 都是 `.cu`，include 不同，不要混在一个目录。

### 现有 kernel

| 路径 | 内容 |
|---|---|
| [`kernels/cuda/layernorm.cu`](kernels/cuda/layernorm.cu) | LayerNorm（Two-Pass Baseline） |
| [`kernels/cuda/welfordLayerNorm.cu`](kernels/cuda/welfordLayerNorm.cu) | LayerNorm（Welford 算法 Single-Pass / Chan's 并行归约） |
| [`kernels/cuda/rmsnorm.cu`](kernels/cuda/rmsnorm.cu) | RMSNorm |
| [`kernels/cuda/transpose.cu`](kernels/cuda/transpose.cu) | float4 向量化矩阵转置（Shared Memory 消除 Bank Conflict） |
| [`kernels/cuda/scan.cu`](kernels/cuda/scan.cu) | inclusive / exclusive scan（单 Block 共享内存与多 Block 三段式流水线） |
| [`kernels/cuda/quant/int8_per_token.cu`](kernels/cuda/quant/int8_per_token.cu) | int8 per-token quant |
| [`kernels/cute/gemm/bf16_tn_sm80.cu`](kernels/cute/gemm/bf16_tn_sm80.cu) | SM80 BF16 TN GEMM |
| [`kernels/triton/layernorm.py`](kernels/triton/layernorm.py) | LayerNorm |
| [`kernels/triton/rmsnorm.py`](kernels/triton/rmsnorm.py) | RMSNorm |
| [`kernels/triton/softmax.py`](kernels/triton/softmax.py) | softmax |
| [`kernels/triton/flashattention.py`](kernels/triton/flashattention.py) | FlashAttention forward |
| [`kernels/triton/scan.py`](kernels/triton/scan.py) | inclusive scan（`tl.cumsum` 实现） |
| [`kernels/triton/quant/int8_per_token.py`](kernels/triton/quant/int8_per_token.py) | int8 per-token quant |
| [`kernels/triton/quant/int8_per_group.py`](kernels/triton/quant/int8_per_group.py) | int8 per-group quant |

## 怎么加新 kernel

1. 先按栈选目录：`kernels/cuda/`、`kernels/cute/` 或 `kernels/triton/`。
2. 该算子只有一份实现时，文件直接放在栈目录下，例如 [`kernels/triton/softmax.py`](kernels/triton/softmax.py)。
3. 会有多个变体再建模：`quant/`、`gemm/`。
4. CuTe / CUDA 文件名：`{dtype}_{layout}_{arch}.cu`，例如 `bf16_tn_sm80.cu`。不要再开 `sm80/` 目录。
5. Triton 一般不写 arch。一个 `.py` 里一个 `@triton.jit`。
6. 不要为 TiledMMA / Copy 提前抽公共头；教学模板要单文件可读。

## LSP 配置

| 文件 | 作用 |
|---|---|
| [`pyrightconfig.json`](pyrightconfig.json) | Pylance：`extraPaths` 指向 vendored Triton；`include` 含 `kernels/`、`examples/` |
| [`.clangd`](.clangd) + [`compile_flags.txt`](compile_flags.txt) | `kernels/cuda`、`examples/cuda` 用 Clang CUDA 前端；CuTe 仍当 C++ 解析 |
| [`.vscode/settings.json`](.vscode/settings.json) | 解释器、clangd、关掉 Microsoft C++ IntelliSense |

`compile_flags.txt` 的 `-I` 相对仓库根，kernel 放在子目录里仍然能跳转到 CUTLASS。

默认版本（可用环境变量覆盖）：Triton `v3.4.0`，CUTLASS `v4.7.0`，CUDA 12 头文件来自 Linux wheel。

NVIDIA 头文件是专有许可，`third_party/cuda-toolkit/`、`cutlass/`、`triton/` 已在 `.gitignore` 里，不要提交。
