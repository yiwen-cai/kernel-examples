#!/usr/bin/env bash
# Bootstrap Mac LSP deps for Triton / CUDA / CuTe (headers + Python source only).
# Does NOT install CUDA Toolkit or compile kernels.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRITON_REF="${TRITON_REF:-v3.4.0}"
# CUTLASS release tags use a leading "v" (for example v4.7.0).
CUTLASS_REF="${CUTLASS_REF:-v4.7.0}"
CUDA_WHEEL_PLATFORM="${CUDA_WHEEL_PLATFORM:-x86_64-manylinux_2_17}"
PYTHON_VERSION_FOR_WHEELS="${PYTHON_VERSION_FOR_WHEELS:-3.12}"

CUDA_SITE="$ROOT/.cache/cuda-site"
CUDA_TK="$ROOT/third_party/cuda-toolkit"
CUTLASS_DIR="$ROOT/third_party/cutlass"
TRITON_DIR="$ROOT/third_party/triton"

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"; }

need uv
need git
need python3

log "uv sync (Python >=3.12, numpy + torch)"
uv sync --python 3.12

mkdir -p "$CUDA_SITE" "$CUDA_TK/include" "$CUDA_TK/bin"

# ---------------------------------------------------------------------------
# CUDA headers from Linux NVIDIA wheels (headers only; not committed)
# uv has no `pip download`; install Linux wheels into a scratch --target tree.
# ---------------------------------------------------------------------------
log "install CUDA component wheels into scratch tree (Linux platform for headers)"
rm -rf "$CUDA_SITE"
mkdir -p "$CUDA_SITE"
uv pip install \
  --python-platform "$CUDA_WHEEL_PLATFORM" \
  --python-version "$PYTHON_VERSION_FOR_WHEELS" \
  --only-binary=:all: \
  --no-deps \
  --target "$CUDA_SITE" \
  nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvcc-cu12 \
  nvidia-cuda-cccl-cu12

log "merge top-level NVIDIA package include/ trees (skip nested libcxx copies)"
# Only merge package-root include dirs, e.g.:
#   nvidia/cuda_runtime/include
#   nvidia/cuda_nvcc/include
#   nvidia/cuda_cccl/include
# Do NOT merge nested paths like cuda/std/detail/libcxx/include — they poison
# Apple libc++ when dumped into the fake toolkit include root.
found_include=0
shopt -s nullglob
for inc_dir in \
  "$CUDA_SITE"/nvidia/*/include \
  "$CUDA_SITE"/nvidia/cu12/include \
  "$CUDA_SITE"/nvidia/cu13/include
do
  [[ -d "$inc_dir" ]] || continue
  # Skip nvvm and other nested tool includes that are not the package root.
  case "$inc_dir" in
    */nvvm/include) continue ;;
  esac
  log "merge ${inc_dir#$CUDA_SITE/}"
  rsync -a "$inc_dir"/ "$CUDA_TK/include/"
  found_include=1
done
shopt -u nullglob
(( found_include == 1 )) || die "no include/ directories found under $CUDA_SITE/nvidia"

if [[ -d "$CUDA_SITE/nvidia/cuda_nvcc/bin" ]]; then
  rsync -a "$CUDA_SITE/nvidia/cuda_nvcc/bin"/ "$CUDA_TK/bin/" || true
fi

# Clang's CUDA wrapper always #include's this; runtime wheels do not ship it.
if [[ -f "$ROOT/third_party/lsp/curand_mtgp32_kernel.h" ]]; then
  cp "$ROOT/third_party/lsp/curand_mtgp32_kernel.h" "$CUDA_TK/include/curand_mtgp32_kernel.h"
fi

# Optional: makes --cuda-path look like a real toolkit if -nocudalib is dropped.
if [[ -f "$CUDA_SITE/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc" ]]; then
  mkdir -p "$CUDA_TK/nvvm/libdevice"
  cp "$CUDA_SITE/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc" \
    "$CUDA_TK/nvvm/libdevice/libdevice.10.bc"
fi

if [[ ! -f "$CUDA_TK/include/cuda.h" ]]; then
  log "cuda.h missing from wheels; writing forward stub"
  cat >"$CUDA_TK/include/cuda.h" <<'EOF'
/* Forward stub for LSP when driver API headers are absent from wheels. */
#pragma once
#include <cuda_runtime.h>
EOF
fi

# Touch a dummy nvcc so --cuda-path layout checks are happier if ever used.
if [[ ! -x "$CUDA_TK/bin/nvcc" ]]; then
  mkdir -p "$CUDA_TK/bin"
  cat >"$CUDA_TK/bin/nvcc" <<'EOF'
#!/bin/sh
echo "stub nvcc for LSP only; not a real compiler" >&2
exit 1
EOF
  chmod +x "$CUDA_TK/bin/nvcc"
fi

# ---------------------------------------------------------------------------
# CUTLASS / CuTe headers (sparse clone)
# ---------------------------------------------------------------------------
if [[ -d "$CUTLASS_DIR/.git" ]]; then
  log "update CUTLASS ($CUTLASS_REF)"
  git -C "$CUTLASS_DIR" fetch --depth 1 origin "refs/tags/$CUTLASS_REF:refs/tags/$CUTLASS_REF" 2>/dev/null \
    || git -C "$CUTLASS_DIR" fetch --depth 1 origin "$CUTLASS_REF"
  git -C "$CUTLASS_DIR" checkout -f "$CUTLASS_REF"
  git -C "$CUTLASS_DIR" sparse-checkout set include
else
  log "sparse-clone CUTLASS ($CUTLASS_REF)"
  rm -rf "$CUTLASS_DIR"
  git clone --depth 1 --filter=blob:none --sparse \
    --branch "$CUTLASS_REF" \
    https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
  git -C "$CUTLASS_DIR" sparse-checkout set include
fi

# ---------------------------------------------------------------------------
# Triton Python package source (sparse clone; no compile)
# ---------------------------------------------------------------------------
if [[ -d "$TRITON_DIR/.git" ]]; then
  log "update Triton ($TRITON_REF)"
  git -C "$TRITON_DIR" fetch --depth 1 origin "refs/tags/$TRITON_REF:refs/tags/$TRITON_REF" 2>/dev/null \
    || git -C "$TRITON_DIR" fetch --depth 1 origin "$TRITON_REF"
  git -C "$TRITON_DIR" checkout -f "$TRITON_REF"
  git -C "$TRITON_DIR" sparse-checkout set python/triton third_party/nvidia/language
else
  log "sparse-clone Triton ($TRITON_REF)"
  rm -rf "$TRITON_DIR"
  git clone --depth 1 --filter=blob:none --sparse \
    --branch "$TRITON_REF" \
    https://github.com/triton-lang/triton.git "$TRITON_DIR"
  git -C "$TRITON_DIR" sparse-checkout set python/triton third_party/nvidia/language
fi

# setup.py copies this into language/extra/cuda; do the same for LSP (symlink).
nvidia_cuda_lang="$TRITON_DIR/third_party/nvidia/language/cuda"
extra_dir="$TRITON_DIR/python/triton/language/extra"
extra_cuda="$extra_dir/cuda"
if [[ -d "$nvidia_cuda_lang" ]]; then
  ln -sfn "../../../../third_party/nvidia/language/cuda" "$extra_cuda"
  log "linked language/extra/cuda -> third_party/nvidia/language/cuda"
fi

# Static extra.cuda import so Pylance can see tl.extra.cuda (pkgutil is dynamic).
if [[ -f "$ROOT/third_party/lsp/triton_extra_init.py" && -d "$extra_dir" ]]; then
  cp "$ROOT/third_party/lsp/triton_extra_init.py" "$extra_dir/__init__.py"
  log "installed static language/extra/__init__.py for LSP"
fi

# JITFunction.__call__ raises unconditionally, which makes tl.cdiv / .max look like
# NoReturn to Pylance. Guard the raise with TYPE_CHECKING.
jit_py="$TRITON_DIR/python/triton/runtime/jit.py"
if [[ -f "$jit_py" ]]; then
  python3 - "$jit_py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
text = p.read_text()
if "TYPE_CHECKING" not in text.split("from typing", 1)[-1].split("\n", 1)[0]:
    text = text.replace(
        "from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overload, Dict, Any, Tuple",
        "from typing import Callable, Generic, Iterable, Optional, TypeVar, Union, overload, Dict, Any, Tuple, TYPE_CHECKING",
        1,
    )
old = '''    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")
'''
new = '''    def __call__(self, *args, **kwargs):
        if TYPE_CHECKING:
            return self.fn(*args, **kwargs)
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")
'''
if old in text:
    p.write_text(text.replace(old, new, 1))
    print("patched JITFunction.__call__ for LSP")
elif "if TYPE_CHECKING:" in text and "Cannot call @triton.jit" in text:
    print("JITFunction.__call__ already patched")
else:
    raise SystemExit(f"failed to patch {p}")
PY
fi

# Minimal triton._C stub so editors/runtime import chains do not die on missing .so
log "write triton._C LSP stubs into vendored tree"
mkdir -p "$TRITON_DIR/python/triton/_C"
cat >"$TRITON_DIR/python/triton/_C/__init__.py" <<'EOF'
# Auto-generated by scripts/bootstrap_lsp.sh — not a real libtriton.
from . import libtriton as libtriton
EOF
cat >"$TRITON_DIR/python/triton/_C/libtriton.py" <<'EOF'
# Auto-generated by scripts/bootstrap_lsp.sh — placeholders for LSP / import.
class _PropagateNan:
    NONE = 0
    ALL = 1

class _IR:
    PROPAGATE_NAN = _PropagateNan

ir = _IR()
passes = object()
interpreter = object()
gluon_ir = object()

def get_cache_invalidating_env_vars():
    return {}
EOF

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
log "verify LSP assets"
checks=(
  "$TRITON_DIR/python/triton/language/__init__.py"
  "$TRITON_DIR/python/triton/language/extra/cuda/libdevice.py"
  "$CUDA_TK/include/cuda_runtime.h"
  "$CUTLASS_DIR/include/cute/tensor.hpp"
  "$ROOT/third_party/lsp/cuda_keywords.h"
  "$CUDA_TK/include/curand_mtgp32_kernel.h"
)
for f in "${checks[@]}"; do
  [[ -f "$f" ]] || die "missing required file: $f"
  printf '  ok %s\n' "${f#$ROOT/}"
done

log "done"
cat <<EOF

Next steps:
  1. In Cursor, select interpreter: $ROOT/.venv/bin/python
  2. Reload the window
  3. Open examples/triton/smoke_add.py, examples/cuda/smoke_add.cu,
     examples/cute/smoke_tensor.cu — Go to Definition on tl.load /
     cudaMalloc / cute::make_tensor
     (examples/ avoids shadowing the real 'triton' package name)

Notes:
  - NVIDIA headers under third_party/cuda-toolkit are not committed (proprietary).
  - This environment is for LSP only; kernels will not compile on macOS.
EOF
