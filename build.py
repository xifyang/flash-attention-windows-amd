"""Build the FlashAttention-2 CK wheel for Windows and AMD RDNA4 GPUs (README.md).

The source is the submodule flash-attention. The wheel has only the forward kernels of
flash_attn_func without dropout, without and with the causal mask (KERNEL_FILTERS), for gfx1200 and
gfx1201 and head_dim 64 and 128, so the build compiles 35 units and not about 960. It links
the torch of this environment.

    python build.py [--out dist]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools._distutils._msvccompiler import _get_vc_env

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "flash-attention"
# The revision of the wheel for one source commit: its local version is ck.<commit>.r<REVISION>.
# 1 (the release v2.8.4-ck.a369df70) had no causal kernels; 2 adds them.
REVISION = 2
# The --filter of CK's generate.py for each API: the forward kernels of batch mode without bias and
# dropout, without mask (_nmask_) and with the mask (_mask_) that flash_attn_func(causal=True) uses:
# setup.py generates the simplified mask of CK, whose kernels take the causal mask at run time. The
# other APIs (backward, split-KV, append-KV) get no kernels.
KERNEL_FILTERS = {
    "fwd": "*_batch_*_nbias_*mask_lse_ndropout_*",
    "fwd_splitkv": "none@none",
    "fwd_appendkv": "none",
    "bwd": "none@none@none",
}
# setup.py with the filters added to its calls of generate.py: setup.py has no option for them.
SETUP_WITH_FILTERS = """
import json, os, runpy, subprocess, sys
filters, run = json.loads(os.environ["KERNEL_FILTERS"]), subprocess.run
def run_with_filter(command, *args, **kwargs):
    if str(command[1]).endswith("generate.py"):
        command = [*command, "--filter", filters[command[command.index("-d") + 1]]]
    return run(command, *args, **kwargs)
subprocess.run = run_with_filter
sys.argv = ["setup.py", *sys.argv[1:]]
runpy.run_path("setup.py", run_name="__main__")
"""


def main() -> None:
    """Check out the submodule (only its CK submodule) and build the wheel into --out."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    out = parser.parse_args().out.resolve()

    git = ["git", "-C", str(SOURCE), "-c", "core.longpaths=true"]
    # --checkout overrides update = none of .gitmodules.
    subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "submodule",
            "update",
            "--init",
            "--checkout",
            "flash-attention",
        ],
        check=True,
    )
    subprocess.run([*git, "submodule", "update", "--init", "csrc/composable_kernel"], check=True)
    commit = subprocess.check_output([*git, "rev-parse", "--short=8", "HEAD"], text=True).strip()
    # setup.py compiles every build/fmha_*.cpp, also those of an earlier build.
    shutil.rmtree(SOURCE / "build", ignore_errors=True)

    rocm_sdk = Path(sys.executable).with_name("rocm-sdk.exe")
    sdk = Path(subprocess.check_output([str(rocm_sdk), "path", "--root"], text=True).strip())
    env = {**os.environ, **{key.upper(): value for key, value in _get_vc_env("x64").items()}}
    # ninja of this environment first: without ninja, torch splits the include paths with spaces.
    path = (Path(sys.executable).parent, sdk / "lib/llvm/bin", sdk / "bin")
    env["PATH"] = os.pathsep.join([*map(str, path), env["PATH"]])
    if shutil.which("ninja", path=env["PATH"]) is None:
        raise RuntimeError("没有 ninja：先 uv pip install ninja psutil wheel")
    env.update(
        BUILD_TARGET="rocm",
        ROCM_HOME=str(sdk),
        ROCM_PATH=str(sdk),
        HIP_PATH=str(sdk),
        # The SDK has the device libraries in lib/llvm/amdgcn/bitcode, not in amdgcn/bitcode.
        HIP_DEVICE_LIB_PATH=str(sdk / "lib/llvm/amdgcn/bitcode"),
        # setup.py takes the MSVC environment of env.
        DISTUTILS_USE_SDK="1",
        GPU_ARCHS="gfx1200;gfx1201",
        OPT_DIM="64,128",
        # setup.py takes 5 GB for each job (2 jobs with 52 GB free); a job takes about 1 GB.
        MAX_JOBS=str(os.cpu_count()),
        FLASH_ATTN_LOCAL_VERSION=f"ck.{commit}.r{REVISION}",
        # Build; do not download the wheel of the release.
        FLASH_ATTENTION_FORCE_BUILD="TRUE",
        KERNEL_FILTERS=json.dumps(KERNEL_FILTERS),
    )
    # The objects go to a short temporary directory: under <source>/build their paths are longer
    # than the 260 characters of Windows. It is set in a configuration file (DIST_EXTRA_CONFIG):
    # with a build command before bdist_wheel, setuptools compiles everything two times.
    with tempfile.TemporaryDirectory(prefix="fa2ck") as temporary:
        config = Path(temporary) / "setup.cfg"
        config.write_text(f"[build]\nbuild_temp = {temporary}\n", encoding="utf-8")
        env["DIST_EXTRA_CONFIG"] = str(config)
        command = [sys.executable, "-c", SETUP_WITH_FILTERS, "bdist_wheel", "--dist-dir", str(out)]
        subprocess.run(command, cwd=SOURCE, env=env, check=True)


if __name__ == "__main__":
    main()
