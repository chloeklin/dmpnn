# Experiment 2: Within-Group Architecture Effect Size

## Methodology

For each matched group (A, B, f_A, f_B), compute:
- Δy = y_individual - group_mean(y)

This isolates the architecture-dependent component from monomer/composition effects.

## Distribution Summary

| Statistic | EA (eV) | IP (eV) |
|-----------|---------|--------|
| Mean Δy | 0.0000 | -0.0000 |
| Std(Δy) | 0.0594 | 0.0582 |
| Mean |Δy| | 0.0368 | 0.0353 |
| Median |Δy| | 0.0211 | 0.0185 |
| P25 |Δy| | 0.0082 | 0.0071 |
| P75 |Δy| | 0.0465 | 0.0436 |
| P95 |Δy| | 0.1302 | 0.1330 |
| Max |Δy| | 0.6102 | 0.5755 |

## Per-Architecture Mean Deviation

| Architecture | EA mean Δ (eV) | EA std | IP mean Δ (eV) | IP std |
|--------------|----------------|--------|----------------|--------|
| alternating | -0.0585 | 0.0756 | +0.0597 | 0.0708 |
| block | +0.0317 | 0.0568 | -0.0356 | 0.0532 |
| random | -0.0122 | 0.0285 | +0.0157 | 0.0264 |

## Label Noise Estimate

**No independent label noise estimate is available.** This is computational (DFT) data, not experimental. The dataset contains one value per (A, B, f_A, f_B, arch) tuple — no replicates exist to estimate noise directly.

However, DFT calculations for EA/IP typically have systematic errors of 0.1–0.3 eV relative to experiment, but **internal consistency** (precision between calculations at the same level of theory) is much better, likely < 0.01 eV.

## Interpretation

- Architecture deviations have std = 0.0594 eV (EA) and 0.0582 eV (IP)
- Mean absolute deviations = 0.0368 eV (EA), 0.0353 eV (IP)
- P95 = 0.1302 eV (EA), 0.1330 eV (IP)

**Architecture effect is substantial** (std > 0.05 eV for both targets). These deviations are well above expected DFT internal precision (< 0.01 eV). This confirms that architecture introduces a real, learnable signal, not merely noise.
