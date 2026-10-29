"""D-1 equivalence machine verification for bitwise_not NPU redesign.

Verifies the three adopted optimization/substitution items of DESIGN.md
section 1.6.1 against the GPU-source baseline formulas, on torch CPU with
exact integer/bool semantics (bitwise ops are exact -- no fp rounding is
involved anywhere, so the reference is the exact integer semantic
``torch.bitwise_not`` on the native dtype; ulp/violation counting is
element-exact equality).

Adopted items (DESIGN.md section 1.6.1):
  A1  int8/int16/int32/int64: GPU ``x ^ (-1)``  ->  NPU ``T.vnot`` (~x)
      Identity: two's complement ``~x == x XOR (2^w - 1)`` and ``-1``
      stores as all-ones (2^w - 1) for width w.
  A2  uint8: GPU ``x ^ 255`` ->  NPU int8-bit-view + ``T.vnot`` + view-back
      Identity: uint8/int8 share the same byte representation; vnot is a
      bitwise op independent of the signedness interpretation.
  A3  bool: GPU ``x == 0`` ->  NPU int8-bit-view + ``T.vxor(x, 1)`` + view-back
      Identity: torch.bool storage contract is {0, 1}; on {0,1}:
      0^1=1 (True), 1^1=0 (False) == logical not == ``x == 0``.

Run:  python3 verify_equiv.py   (exit 0 = all EQUIV_PASS)
"""

import sys

import torch

FAILURES = []


def report(item, dtype, n_points, n_viol, detail):
    verdict = "EQUIV_PASS" if n_viol == 0 else "EQUIV_FAIL"
    print(
        f"| {item} | {dtype} | {n_points} | {n_viol} | {detail} | {verdict} |"
    )
    if n_viol != 0:
        FAILURES.append((item, dtype, detail))


def corners_int(signed, bits):
    """Value corners: 0, +-1, extrema, alternating nibble patterns."""
    if signed:
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        lo, hi = 0, 2 ** bits - 1
    half = 2 ** bits // 2
    vals = [0, 1, hi, lo, hi // 3, lo // 3 if signed else half + half // 3]
    pat1, pat2 = 0, 0
    for k in range(bits):
        if k % 2 == 0:
            pat1 |= 1 << k
        else:
            pat2 |= 1 << k
    vals.append(pat1 if not signed or pat1 <= hi else pat1 - 2 ** bits)
    vals.append(pat2 if not signed or pat2 <= hi else pat2 - 2 ** bits)
    return vals


def main():
    torch.manual_seed(0)
    n_rag, n_div = 1000, 1024  # ragged tail + exact tile counts

    print("## D-1 verification table (bitwise exact; ref = torch.bitwise_not)")
    print("| item | dtype | points | violations | detail | verdict |")
    print("|------|-------|--------|------------|--------|---------|")

    # ------------------------------------------------------------------
    # A1: int8/int16/int32/int64 -- x ^ (-1) == bitwise_not(x)
    # ------------------------------------------------------------------
    for name, bits in (("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)):
        dt = getattr(torch, name)
        lo, hi = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        rand = torch.randint(lo, hi, (n_div,), dtype=torch.int64).to(dt)
        rand2 = torch.randint(lo, hi, (n_rag,), dtype=torch.int64).to(dt)
        x = torch.cat([rand, rand2, torch.tensor(corners_int(True, bits), dtype=dt)])
        baseline = torch.bitwise_xor(x, -1)  # GPU source formula (scalar -1)
        candidate = torch.bitwise_not(x)  # NPU T.vnot semantic (~x)
        viol = int((baseline != candidate).sum())
        report("A1 xor(-1)->vnot", name, x.numel(), viol, "exact bitwise identity")

    # ------------------------------------------------------------------
    # A2: uint8 -- x ^ 255 == bitwise_not(x) == int8-view bitwise_not roundtrip
    # ------------------------------------------------------------------
    u8 = torch.cat(
        [
            torch.randint(0, 256, (n_div,), dtype=torch.uint8),
            torch.randint(0, 256, (n_rag,), dtype=torch.uint8),
            torch.tensor(corners_int(False, 8), dtype=torch.uint8),
        ]
    )
    baseline_u8 = torch.bitwise_xor(u8, torch.tensor(255, dtype=torch.uint8))
    golden_u8 = torch.bitwise_not(u8)
    viol_a = int((baseline_u8 != golden_u8).sum())
    report("A2a xor(255)->not", "uint8", u8.numel(), viol_a, "GPU formula vs vnot")

    view_route = torch.bitwise_not(u8.view(torch.int8)).view(torch.uint8)
    viol_b = int((view_route != golden_u8).sum())
    report("A2b int8-view route", "uint8", u8.numel(), viol_b, "NPU route vs vnot")

    # ------------------------------------------------------------------
    # A3: bool -- storage {0,1}: (x==0) == bitwise_not(x) == xor-1 route
    # ------------------------------------------------------------------
    b = torch.cat(
        [
            (torch.rand(n_div) > 0.5),
            (torch.rand(n_rag) > 0.5),
            torch.tensor([False, True, False, True], dtype=torch.bool),
        ]
    )
    golden_b = torch.bitwise_not(b)
    gpu_route = (b.view(torch.int8) == 0)  # GPU x==0 formula
    viol_a = int((gpu_route != golden_b).sum())
    report("A3a x==0 baseline", "bool", b.numel(), viol_a, "GPU formula vs not")

    npu_route = torch.bitwise_xor(b.view(torch.int8), torch.tensor(1, dtype=torch.int8)).view(
        torch.bool
    )
    viol_b = int((npu_route != golden_b).sum())
    report("A3b xor(1) route", "bool", b.numel(), viol_b, "NPU route vs not")

    # out-of-contract note (NOT an equivalence claim): storage value 2
    rogue = torch.tensor([2], dtype=torch.int8)
    print(
        "note: bool storage outside {0,1} is outside the torch.bool contract "
        f"(xor route gives {int(rogue ^ 1)} -> truthy view; GPU x==0 gives 0); "
        "workload/test inputs are contract-valid {0,1} only (gen_inputs rand>0.5)."
    )

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} EQUIV_FAIL item(s) -- ADOPTION FORBIDDEN")
        sys.exit(1)
    print("RESULT: all adopted items EQUIV_PASS (exact, 0 violations)")


if __name__ == "__main__":
    main()
