#!/usr/bin/env python3
"""Generate DP16 analytic inputs using the shared DP8 input-generation code."""

from run_inference_dp8 import ROOT, main

SEQUENCES_CSV = ROOT / "data/eaip_variant/dp16_cls.csv"


if __name__ == "__main__":
    main(dp=16, sequences_csv=SEQUENCES_CSV)
