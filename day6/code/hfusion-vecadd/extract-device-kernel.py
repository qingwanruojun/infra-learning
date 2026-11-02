#!/usr/bin/env python3
"""Extract the device entry from the HFusion pipeline output.

The source-built HFusion pipeline emits both the outlined DEVICE function and a
duplicate HOST wrapper with the original name (e.g. @add). CANN 9.0.0's hivmc
cannot legalize the HIVM ops that remain in that HOST wrapper, so drop only the
host entry before producing the device ELF. The device kernel and its tiling
helper functions (referenced by attributes) are kept.
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--host-func", default="add")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    out = []
    dropped = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("  func.func @"):
            name = line.split("@", 1)[1].split("(", 1)[0]
            j = i + 1
            while j < len(lines) and lines[j] != "  }":
                j += 1
            if j == len(lines):
                sys.exit(f"unterminated function @{name}")
            if name == args.host_func:
                dropped.append(name)
            else:
                out.extend(lines[i : j + 1])
            i = j + 1
        else:
            out.append(line)
            i += 1

    if not dropped:
        sys.exit(f"host function @{args.host_func} not found")
    if not any("hacc.entry" in line and "DEVICE" in line for line in out):
        sys.exit("no DEVICE entry function left")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"kept DEVICE entry, dropped HOST function(s): {', '.join(dropped)}")


if __name__ == "__main__":
    main()
