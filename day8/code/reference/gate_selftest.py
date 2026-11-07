"""Self-test for the acceptance exit gate (no NPU work).

Run:
  python /mnt/workspace/learn/infra-learning/day8/code/reference/gate_selftest.py

Exit 0 only if the gate returns 0 for three passing cases with three caught
negative controls, and 1 when any case or any negative control is wrong.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_acceptance import all_ok, exit_code  # noqa: E402


def case(pass_value, neg_pass_value):
    return {
        "metrics": {"pass": pass_value},
        "negative_control": {"pass": neg_pass_value},
    }


def main():
    good = [case(True, False) for _ in range(3)]
    one_bad_case = [case(True, False), case(False, False), case(True, False)]
    one_missed_control = [case(True, False), case(True, False), case(True, True)]

    checks = [
        ("all good", good, True, 0),
        ("one precision fail", one_bad_case, False, 1),
        ("one negative control missed", one_missed_control, False, 1),
    ]
    ok = True
    for label, results, want_ok, want_code in checks:
        got_ok = all_ok(results)
        got_code = exit_code(results)
        this_ok = got_ok == want_ok and got_code == want_code
        ok &= this_ok
        print(
            f"{'PASS' if this_ok else 'FAIL'} {label}: "
            f"all_ok={got_ok} (want {want_ok}), "
            f"exit_code={got_code} (want {want_code})"
        )
    print("GATE_SELFTEST_PASS" if ok else "GATE_SELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
