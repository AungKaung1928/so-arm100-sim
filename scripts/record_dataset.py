"""Thin wrapper: the recorder lives in the package, see so_arm100_sim/record.py.

    python scripts/record_dataset.py --root data/lift_red --repo-id local/so_arm100_lift_red \
        --tasks lift:red --episodes 50 --cameras front
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from so_arm100_sim.record import main   # noqa: E402

if __name__ == "__main__":
    main()
