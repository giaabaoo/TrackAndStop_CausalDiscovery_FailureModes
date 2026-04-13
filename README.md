# Failure Modes of Adaptive Online Experimental Design for Causal Discovery

Empirical study of where the Track-and-Stop (TsP) algorithm breaks down. We run controlled experiments varying key assumptions and measure Structural Hamming Distance (SHD) against a random intervention baseline.

## Experiments

**Table 1 — Signal strength** (`run_table1.py`)  
Sweep graph density ∈ {0.3, 0.5, 0.7}. Fixed: n=5, B=5000, 50 trials.

**Table 2 — Unobserved confounding** (`run_table2.py`)  
Sweep num\_latents ∈ {0,1,2,3,4,5}. Latents shift P(child=1|parents) by ±δ=0.3 with fanout=2. Fixed: n=5, density=0.5, B=5000, 50 trials.

**Table 3 — Noisy interventions** (`run_table3.py`)  
Sweep noise level ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}. Fixed: n=5, density=0.5, B=5000, 50 trials.

**Table 4 — Sample budget** (`run_table4.py`)  
Sweep B ∈ {500, 1000, 2000, 5000, 10000}. Fixed: n=5, density=0.5, 50 trials.

## Key Findings

- TsP is robust to graph density and sample budget variations.
- TsP collapses immediately under unobserved confounding: SHD jumps ~6× at 1 latent node while Random is barely affected.
- Noisy interventions degrade TsP more than Random at higher noise levels.

## Structure

```
run_table1-4.py          # experiment scripts
TsP.py                   # Track-and-Stop algorithm (Harsh's version, with latent confounder support)
Code_with_Instructions/  # original codebase from authors
figures/                 # generated plots
report.tex               # paper draft
```

## Reference

Kocaoglu, M., Shanmugam, K., & Bareinboim, E. (2017).  
**Experimental Design for Learning Causal Graphs with Latent Variables.**  
*Advances in Neural Information Processing Systems (NeurIPS)*, 30.

Maiti, A., Bhattacharyya, C., & Saha, A. (2024).  
**Adaptive Online Experimental Design for Causal Discovery.**  
*International Conference on Machine Learning (ICML)*.
