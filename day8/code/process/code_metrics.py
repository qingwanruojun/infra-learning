"""Static structure metrics for the Day 3 and Day 8 RMSNorm implementations."""

import ast
import json
import re
import sys


def metrics(path):
    src = open(path).read()
    tree = ast.parse(src)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    tile_apis = sorted(set(re.findall(r"\bT\.([A-Za-z_][A-Za-z0-9_]*)", src)))
    return {
        "path": path,
        "lines_total": src.count("\n") + (0 if src.endswith("\n") else 1),
        "functions": funcs,
        "n_functions": len(funcs),
        "tile_apis": tile_apis,
        "n_alloc": len(re.findall(r"\balloc_\w+\(", src))
        + len(re.findall(r"(?<![\w.])alloc\(", src)),
        "n_parallel": len(re.findall(r"\bT\.Parallel\(", src)),
        "n_v_prefix_ops": len(re.findall(r"\bT\.v[a-z]+\(", src)),
        "n_manual_sync": len(re.findall(r"T\.(barrier|set_flag|wait_flag)", src)),
        "n_tests": len(re.findall(r"^def test_", src, re.M)),
        "n_assert": len(re.findall(r"\bassert\b", src)),
        "has_pytest": "pytest" in src,
        "has_main": "__main__" in src,
        "has_benchmark": "def benchmark" in src,
        "host_padding": bool(re.search(r"F\.pad|\.contiguous\(|torch\.cat|torch\.stack", src)),
    }


if __name__ == "__main__":
    out = [metrics(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=2, ensure_ascii=False))
