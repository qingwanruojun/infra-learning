# Copyright (c) Huawei Technologies Co., Ltd. 2026.
"""Tiled amin row-reduction kernel -- Stage-4 perf_opt shim.

The Stage-4 gate requires a single final artifact for both kernels; the
tuned merged module is ``perf_opt/_simple_reduce_kernel.py`` (it contains
the tuned single-tile factory AND the tiled factory, whose baseline form
remained the best after rounds 3a/3b/6a/7b -- see perf_opt/opt_log.md).
This shim forwards the tiled factory (and its own regression entry) so the
wrapper's per-function perf_opt import toggle works unchanged:

    from .amin_fwd_kernel.perf_opt._simple_reduce_kernel_tiled import (
        _simple_reduce_kernel_tiled,
    )
"""

try:
    # package context (wrapper import path)
    from ._simple_reduce_kernel import (  # noqa: F401
        _simple_reduce_kernel_tiled,
        golden_amin as _golden_amin_tiled,
        run_L0_tiled,
        run_L1_tiled,
        run_L2_tiled,
        run_boundary_tiled,
    )
except ImportError:
    # direct-script execution (python perf_opt/_simple_reduce_kernel_tiled.py)
    import importlib.util as _ilu
    import os as _os

    _here = _os.path.dirname(_os.path.abspath(__file__))
    _spec = _ilu.spec_from_file_location(
        "_amin_merged_final", _os.path.join(_here, "_simple_reduce_kernel.py")
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _simple_reduce_kernel_tiled = _mod._simple_reduce_kernel_tiled
    _golden_amin_tiled = _mod.golden_amin
    run_L0_tiled = _mod.run_L0_tiled
    run_L1_tiled = _mod.run_L1_tiled
    run_L2_tiled = _mod.run_L2_tiled
    run_boundary_tiled = _mod.run_boundary_tiled

__all__ = [
    "_simple_reduce_kernel_tiled",
    "run_L0_tiled",
    "run_L1_tiled",
    "run_L2_tiled",
    "run_boundary_tiled",
]


def main():
    # Full tiled regression suite (equivalent to the baseline module's
    # --level all for the tiled kernel).
    run_L0_tiled()
    run_L1_tiled()
    run_L2_tiled()
    run_boundary_tiled()
    print("\033[92mAll check passed!\033[0m")


if __name__ == "__main__":
    main()
