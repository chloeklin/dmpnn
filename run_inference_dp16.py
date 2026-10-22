#!/usr/bin/env python3
"""DP16 inference, sharing the original polymer-chemprop adapter with DP8.

Configure common lookup/checkpoint/code paths in run_inference_dp8.py.
The DP8 sequence path is not used here. All paths also have CLI overrides.
"""

from run_inference_dp8 import main

SEQUENCES_CSV = "[PATH_TO_DP16_CSV]"


if __name__ == "__main__":
    main(dp=16, sequences_csv=SEQUENCES_CSV)
