# FlashAttention-2 CK for Windows and AMD RDNA4 GPUs

Prebuilt wheels of [ROCm/flash-attention](https://github.com/ROCm/flash-attention) (CK backend) for
Windows x64, AMD gfx1200 and gfx1201 (Radeon RX 9060, RX 9070), and the
[AMD ROCm PyTorch](https://repo.amd.com/rocm/whl-multi-arch/) of each release. The upstream project
publishes no Windows wheels.

The wheels have only the forward kernels of `flash_attn_func` without dropout, for head_dim 64 and
128; since `r2` without and with the causal mask (`causal=True`). The backward pass,
variable-length, KV-cache and split-KV calls are in the package but have no kernels: they raise at
run time.

| Release | flash-attention | composable_kernel | torch | Python | Kernels |
|---|---|---|---|---|---|
| `v2.8.4-ck.a369df70.r2` | `ck_improve_main` a369df70 | c56c6750 | 2.11.0+rocm7.14.0 | 3.12 | without and with `causal=True` |
| `v2.8.4-ck.a369df70` | `ck_improve_main` a369df70 | c56c6750 | 2.11.0+rocm7.14.0 | 3.12 | without `causal=True` |

A wheel works only with the torch of its row.

## Use

```toml
"flash-attn @ https://github.com/xifyang/flash-attention-windows-amd/releases/download/v2.8.4-ck.a369df70.r2/flash_attn-2.8.4%2Bck.a369df70.r2-cp312-cp312-win_amd64.whl ; sys_platform == 'win32'",
```

```python
from flash_attn import flash_attn_func

output = flash_attn_func(query, key, value)  # [B, S, H, D], float16 or bfloat16
output = flash_attn_func(query, key, value, causal=True)  # r2 and later
```

## Build

On Windows with git, the Visual Studio x64 build tools, and a Python environment that has the torch
of the table:

```bat
git clone https://github.com/xifyang/flash-attention-windows-amd
cd flash-attention-windows-amd
pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ rocm-sdk-devel==7.14.0
pip install ninja psutil wheel
python build.py
```

`build.py` checks out the submodule `flash-attention` and only its submodule `csrc/composable_kernel`
(the other submodules have paths that are too long for Windows; do not clone with `--recursive`),
and writes the wheel to `dist`. It adds `--filter` to the calls of CK's `generate.py` in
`setup.py`, so it compiles 35 units (24 of them kernels: 12 without and 12 with the mask of
`causal=True`) instead of about 960 (3 to 5 minutes on 24 cores with the clone). The local version of
the wheel is `ck.<commit>.r<REVISION>`: `REVISION` in `build.py` counts the kernel sets of one
source commit.

## License

The wheels contain flash-attention ([LICENSE-flash-attention](LICENSE-flash-attention), BSD 3-Clause) and
composable_kernel ([LICENSE-composable_kernel](LICENSE-composable_kernel), MIT).
