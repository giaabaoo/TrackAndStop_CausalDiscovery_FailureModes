# Track-and-Stop Causal Discovery — Algorithm Overview

## Problem Setting

Given an unknown DAG (Bayesian network) over binary variables, the goal is to **identify the true causal graph** (recover all edge directions) by performing targeted interventions, minimizing the total number of interventional samples used.

---

## Code Structure

```
Code_with_Instructions/
├── run_me.py              # Entry point — reproduces paper figures
├── results_generator.py   # Orchestrates all algorithms and plots SHD vs samples
├── TsP.py                 # Main algorithm: Track-and-Stop Discovery
├── DCT.py                 # Baseline: Directed Clique Tree (DCT) policy
├── Radpt.py               # Baseline: r-Adaptive separator policy
├── Rnd.py                 # Baseline: Random intervention policy
└── GIES.py                # Baseline: GIES (Greedy Interventional Equivalence Search)
```

---

## How to Run

```bash
python run_me.py
```

`run_me.py` calls `results_generator.Run_CausalDiscovery_Algorithms_and_plot_results(Num_Grphs, nodes, degree, Max_samples, Gap)` for each experiment configuration (fig 3a/b/c, fig 4a/b/c). Parameters:

| Parameter    | Meaning                                          |
|--------------|--------------------------------------------------|
| `Num_Grphs`  | Number of random graphs averaged over (50)       |
| `nodes`      | Number of nodes in each DAG (5–10)               |
| `degree`     | Edge density (0 = empty, 1 = complete)           |
| `Max_samples`| Max interventional samples per graph (100,000)   |
| `Gap`        | Sample resolution for SHD evaluation (2,000)     |

Each algorithm returns `Data_save`: a list of SHD-vs-sample arrays, which are averaged and plotted.

---

## Main Algorithm: Track-and-Stop (`TsP.Track_and_stop`)

**File:** `TsP.py:669`

### Key Idea

Track-and-Stop adapts the *Track-and-Stop* bandit framework to causal discovery. It maintains a probability distribution over observed outcomes under each intervention group, and iteratively selects which intervention group to sample from in order to **minimize KL divergence** between the current empirical distribution and the set of hypothetical distributions under all candidate MPDAGs.

### Setup Phase

1. **Generate graph:** Random chordal DAG via `shanmugam_random_chordal(nodes, degree)`.
2. **Build Bayesian network:** `pyAgrum.fastBN` + random CPTs.
3. **Graph coloring → intervention groups:** Greedy graph coloring (`greedyColoring`) on the undirected skeleton partitions nodes into color classes `I = [I_0, I_1, ...]`. Each color class is one intervention group (simultaneously-intervenable independent nodes).
4. **Build MPDAG:** Start from the true DAG's skeleton (all edges undirected). Apply Meek rules (`MPDAG.apply_meek_rules`) to propagate any known orientations.
5. **Enumerate MPDAGs per intervention group:** For each group `I_k`, orient all edges crossing the cut between `I_k` and the rest in every possible direction → `MPDAG_LIST[k]` (all consistent completions).
6. **Golden intervention config:** For each group, find the intervention value configuration (0/1 for each intervened node) that **maximizes the minimum pairwise Euclidean distance** among causal effect vectors across MPDAGs. This is the most discriminating configuration.
7. **Pre-sample data:** Block-sample 100,000 interventional observations for each group under the golden configuration (`block_sample_intervention`).

### Main Loop (`TsP.py:799`)

```
while samples < Max_samples:
    for each intervention group I_k:
        compute KL(empirical_P || predicted_P_under_each_MPDAG)
        α*[k] = 1 / min_KL   (inverse of easiest-to-confuse alternative)
        d*[k] = argmin MPDAG (most likely true graph from group k's view)
    
    Dstar_E = union of d*[k].arcs across all groups  (current best DAG estimate)
    
    # Forced exploration: ensure each arm sampled ≥ 25√t times
    if min(Nt) < 25√t:
        act = argmin(Nt)   # sample least-sampled group
    else:
        act = argmax(t·α* - Nt)   # tracking rule: balance allocation to α*
    
    sample one observation from group `act`
    update empirical distribution Ps[act]
    record SHD(Truedag, Dstar_E)
    
    if SHD == 0 and t >= 1000:
        stop (correct DAG identified)
```

### Key Functions in `TsP.py`

| Function | Role |
|---|---|
| `shanmugam_random_chordal` | Generate random chordal DAG |
| `greedyColoring` + `indices_of_elements` | Partition nodes into intervention groups |
| `MPDAG` class | Represent partially directed graph; apply Meek rules |
| `OrientCut_and_Enumeratmpdags` | Enumerate all MPDAG completions for each intervention group |
| `enumerate_causaleffects` | Compute P(Y \| do(I=config)) for each MPDAG candidate |
| `IdentifyCausaLEffectinMPDAG` | ID formula via PCO (partial causal ordering) |
| `PCO` | Compute partial causal ordering of ancestors |
| `block_sample_intervention` | Sample from BN under hard intervention |
| `Sample_and_update_dist` | Update empirical count distribution |

---

## Baselines

### DCT — Directed Clique Tree (`DCT.py:927`)
Orients the DAG by intervening on cliques of the chordal graph. Two phases:
- **Phase I:** Intervene on tree centroids iteratively to split the clique tree.
- **Phase II:** Resolve residual unoriented edges inside each clique.
Uses chi-square CI tests (`Learn_cut`) to orient edges from interventional data.

### r-Adaptive (`Radpt.py:804`)
Separator-based adaptive policy. Computes balanced partitions of the clique tree and a minimum vertex cover, then selects the cheaper intervention set per round. Runs for up to `r-1` adaptive rounds.

### Random (`Rnd.py:478`)
Selects intervention groups uniformly at random, using chi-square tests to orient cut edges at each sample checkpoint.

### GIES (`GIES.py`)
Standard batch algorithm (Greedy Interventional Equivalence Search) applied passively — no adaptive sample allocation.

---

## Metric

**SHD (Structural Hamming Distance):** Number of edge additions + deletions + reversals needed to go from estimated DAG to true DAG. Plotted vs. cumulative interventional samples.
