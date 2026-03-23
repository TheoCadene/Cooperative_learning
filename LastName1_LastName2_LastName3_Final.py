#!/usr/bin/env python3
"""
Cooperative Kernel Regression — final runnable script (project PDF + plots).
Regenerates all figures in ./figures/ as .pdf files.
Requires: numpy, matplotlib, pickle (stdlib).
"""
import os
import sys

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.getcwd())

    import main_part1
    import main_part1_robustness
    import main_part2
    import main_part3

    print("Running Part I (distributed algorithms)...")
    main_part1.main()
    print("Running Part I (robustness / scaling)...")
    main_part1_robustness.main()
    print("Running Part II (FedAvg)...")
    main_part2.main()
    print("Running Part III (DGD-DP)...")
    main_part3.main()
    print("Done. Figures in ./figures/")


if __name__ == "__main__":
    main()
