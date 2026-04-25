# Concepts Guide: Track-and-Stop Causal Discovery

A beginner-friendly explanation of every concept in this project, from the ground up.

---

## 1. What is a Causal Graph?

Imagine you observe that people who carry lighters tend to get lung cancer. Does carrying a lighter *cause* cancer? No — smoking causes both. The lighter is just correlated, not causal.

A **causal graph** (also called a DAG — Directed Acyclic Graph) is a diagram where:
- **Nodes** = variables (e.g., Smoking, Cancer, Lighter)
- **Arrows** = "directly causes" (Smoking → Cancer, Smoking → Lighter)
- **Directed** = arrows have direction (cause → effect)
- **Acyclic** = no cycles (X cannot cause itself, even through a chain)

**Why do we need it?** If you know the causal graph, you can predict what happens when you *intervene* (e.g., force someone to stop smoking) rather than just observe. This is essential for medicine, policy, economics.

---

## 2. What is a Bayesian Network?

A Bayesian Network is a causal graph **plus numbers**. Each node has a **CPT (Conditional Probability Table)** — a table that tells you:

> "Given the values of my parents, what is the probability that I am 0 or 1?"

**Example:** If Smoking ∈ {0,1} and Cancer ∈ {0,1}:
```
P(Cancer=1 | Smoking=0) = 0.01
P(Cancer=1 | Smoking=1) = 0.30
```

In our code, we use binary variables (0 or 1 only) and generate CPTs randomly with `bn.generateCPTs()`.

**Why do we need it?** To simulate fake data (samples) from the true causal model, so we can test if our algorithm recovers the true graph.

---

## 3. What is an Intervention?

**Observation:** You watch what happens naturally. You might see smokers have cancer more often — but maybe they also drink more, live in cities, etc. Confounded.

**Intervention (do-operator):** You *force* a variable to a specific value and see what happens. Like a randomized controlled trial. Written as `do(Smoking=1)`.

Under an intervention on node X:
- You **cut all incoming edges** to X (remove its parents)
- Set X to a fixed value
- Everything downstream of X is still causally affected

This breaks confounding because X is no longer influenced by anything — it's externally controlled.

**In code:** `block_sample_intervention(bn, intervention_set, ...)` removes parent edges and samples from the modified network.

---

## 4. What is Causal Discovery?

**Causal discovery** = learning the causal graph from data, without knowing it in advance.

Two settings:
- **Observational:** You only watch (no interventions). You can learn correlations but often can't tell direction (does A→B or B→A?)
- **Interventional:** You can perform experiments (do(X=1)). This helps resolve ambiguous directions.

Our project is in the **interventional** setting. The algorithm performs experiments and uses the results to figure out the true causal graph.

---

## 5. What is a Chordal Graph?

Before building a causal graph, we need to pick a structure. We use **chordal graphs**.

A graph is chordal if every cycle of length 4+ has a "chord" (a shortcut edge). This is a mathematical property that makes the graph easier to color and decompose.

**Why chordal?** Chordal graphs have nice theoretical properties — they can be efficiently decomposed into cliques (fully connected subgraphs), which makes it easy to design intervention strategies.

**In code:** `shanmugam_random_chordal(nodes, degree)` generates a random chordal DAG.

---

## 6. What is Graph Coloring and Why Do We Use It?

**Graph coloring** = assign colors to nodes such that no two adjacent (connected) nodes share the same color.

**Example:** On a map, color countries so neighboring countries have different colors.

**Why do we use it for interventions?**

If two nodes are not connected (no edge between them), intervening on both at the same time doesn't cause them to interfere — we can intervene on them simultaneously.

So: nodes with the **same color** = not adjacent = can be intervened on together.

This gives us **intervention groups** (called `I` in the code). Each group is one "experiment" we can run.

**In code:** `greedyColoring(graph_dict, ...)` colors the graph. `indices_of_elements(r)` collects nodes of each color into groups.

---

## 7. What is an MPDAG?

When you start causal discovery, you don't know which edges exist or their directions. As you gather data, you partially figure it out.

An **MPDAG (Maximally Partially Directed Acyclic Graph)** represents your current state of knowledge:
- Some edges are **directed** (→) — you're confident about direction
- Some edges are **undirected** (—) — you know they're connected but not which way

As you collect more data, undirected edges become directed.

**Meek rules** are logical rules that propagate known orientations. For example: if A→B—C and making C→B would create a cycle, then B→C must be true.

**In code:** The `MPDAG` class stores directed (`Arcs`) and undirected (`Edges`). `apply_meek_rules()` propagates orientations.

---

## 8. What is SHD (Structural Hamming Distance)?

**SHD** measures how wrong your estimated graph is compared to the true graph.

It counts:
- **Missing edges:** True graph has A→B, your estimate has nothing
- **Extra edges:** Your estimate has A→B, true graph has nothing
- **Wrong direction:** True graph has A→B, you estimated B→A

Lower SHD = better. SHD=0 means perfect recovery.

**Why use SHD?** It's simple and directly measures graph accuracy. But it has limits — it doesn't tell you *why* you got it wrong (noise vs. systematic bias).

---

## 9. What is KL Divergence?

**KL Divergence (Kullback-Leibler)** measures how different two probability distributions are.

```
KL(P || Q) = sum over x: P(x) * log(P(x) / Q(x))
```

- KL = 0 means P and Q are identical
- KL > 0 means they differ (higher = more different)
- KL is **not symmetric**: KL(P||Q) ≠ KL(Q||P)

**In our algorithm:** We compare the empirical distribution we observe (P, from samples) against what each candidate MPDAG *predicts* we should see (Q). The MPDAG whose prediction is closest to our observations (lowest KL) is our best guess for the true graph.

**In code:** `rel_entr` from scipy computes element-wise KL terms. `sum(rel_entr(Ps, PD[dindex]))` gives KL for one candidate.

---

## 10. What is the Track-and-Stop Algorithm?

This is the core algorithm of the project. It answers the question:

> "Given a budget of N interventional samples, how should I allocate them across intervention groups to identify the true causal graph as fast as possible?"

### The Bandit Analogy

Think of it like a slot machine problem. You have K slot machines (intervention groups). Each pull gives you information. Which machine should you pull next to identify the best one fastest?

**Track-and-Stop** was originally designed for this bandit problem and adapted here for causal discovery.

### The Two-Part Strategy

**Part 1 — Track:** Compute the "ideal" allocation `α*` — how much fraction of samples should go to each intervention group if you had infinite time. This is computed from KL divergences.

**Part 2 — Stop:** At each step, pick the group where you're *most behind* relative to `α*`:
```
act = argmax(t * α* - N_t)
```
where `N_t[k]` = samples so far from group k, `t` = total samples.

**Forced exploration:** Before trusting `α*`, ensure each group has at least `25√t` samples. Otherwise you might under-explore rare but important groups.

### What α* Means

`α*[k] = 1 / min_KL_k`

The group where the worst-case alternative MPDAG is closest to your observations (small min KL) gets more samples — because you're most uncertain there. It's inversely proportional to how easy it is to get confused.

---

## 11. What is Causal Sufficiency?

**Causal sufficiency** means: all common causes of observed variables are also observed.

In other words — **no hidden confounders**.

Example: If Smoking causes both Cancer and Bad Breath, and Smoking is in your dataset, that's causally sufficient. If Smoking is hidden (you only observe Cancer and Bad Breath), that's a **violation of causal sufficiency**.

**TsP assumes causal sufficiency.** This is a strong assumption that often fails in the real world.

---

## 12. What is a Latent Confounder?

A **latent confounder** (also called hidden common cause) is an unobserved variable that causally affects two or more observed variables.

```
H (hidden)
↓       ↓
A       B
```

You see A and B correlated, but you can't observe H. If you try to learn the causal graph between A and B, you'll be confused — is there a direct edge A→B, B→A, or is it all due to H?

**In our experiment:**
- We add `num_latents` hidden binary nodes (named H0, H1, ...)
- Each connects to `fanout=2` random observed nodes
- H shifts `P(child=1 | parents)` by ±δ=0.3
- During interventions, H's influence remains intact — you can't intervene on what you can't see
- TsP never sees H — it thinks the world is causally sufficient

**In code:** `add_latent_confounders(bn, num_latents, fanout, delta, seed)` builds this corrupted BN.

---

## 13. Why Does TsP Fail Under Confounding?

TsP builds its policy based on KL divergence between observed distributions and what each candidate MPDAG predicts.

When confounders exist:
1. The observed distribution `P_empirical` is **shifted** by the hidden H
2. None of the candidate MPDAGs (which don't include H) predict this shifted distribution
3. TsP picks the "closest" MPDAG, but it's comparing apples to oranges
4. It may **confidently converge to the wrong graph** — not because of noise, but because the model is systematically wrong

Random policy is less affected because it doesn't build a probabilistic model — it just picks uniformly and uses simple chi-square tests. It can't be misled by a model it doesn't have.

---

## 14. What is a Chi-Square Test (used in Random baseline)?

The **chi-square independence test** answers: "Are variables X and Y statistically independent given an intervention?"

If we intervene on a set I and look at the data, we test:
- H0: X and Y are independent in this data
- H1: X and Y are dependent (likely connected)

If they're dependent, there's probably an edge between them.

**In code:** `Learn_cut` in `Rnd.py` runs chi-square tests on the interventional data to orient edges. It indexes data columns **by node ID position**, which is why column ordering matters.

---

## 15. What is a CPT (Conditional Probability Table)?

A CPT specifies a node's distribution given all combinations of its parents' values.

**Example:** Node C with parents A and B:
```
A=0, B=0 → P(C=1) = 0.1
A=0, B=1 → P(C=1) = 0.4
A=1, B=0 → P(C=1) = 0.7
A=1, B=1 → P(C=1) = 0.9
```

The CPT has `2^(num_parents) × 2` entries for binary variables.

**In code:** `bn.cpt(node_id)` accesses a node's CPT. `bn.generateCPTs()` fills all CPTs with random values. Important: call this **once only** — calling it twice regenerates random values, making the algorithm use different CPTs than the sampler (a bug we fixed).

---

## 16. What is the do-Calculus / Causal Effect?

When you intervene `do(X=x)`, the **causal effect** on Y is:
```
P(Y | do(X=x))
```

This is different from `P(Y | X=x)` (observational conditioning), because intervention removes confounding.

**In our algorithm:** `enumerate_causaleffects(MPDAG, V, I, Pv, config)` computes `P(V | do(I=config))` for each candidate MPDAG. These are the "predicted distributions" we compare against empirical data using KL.

---

## 17. What is the Golden Intervention Configuration?

For each intervention group `I_k`, we can set each intervened node to 0 or 1. That's `2^|I_k|` possible configurations.

The **golden configuration** is the one that makes the candidate MPDAGs' predicted distributions as **spread apart as possible** — maximizing the minimum pairwise Euclidean distance between their causal effect vectors.

**Why?** If candidates predict very different distributions under this configuration, a single sample is very informative for distinguishing them. We want maximum discriminating power.

**In code:**
```python
p_search_min[v] = min_row_distance(tmp3)  # min pairwise distance across MPDAGs
golden_intv[k] = argmax(p_search_min)     # config that maximizes this
```

---

## 18. Putting It All Together

Here is the full pipeline end to end:

```
1. Generate random chordal DAG
         ↓
2. Assign random CPTs → Bayesian Network (ground truth)
         ↓
3. Color graph → intervention groups I_0, I_1, ...
         ↓
4. For each group: enumerate all possible MPDAG completions
         ↓
5. Find golden intervention config per group (max discriminability)
         ↓
6. Pre-sample a large pool of interventional data per group
         ↓
7. MAIN LOOP:
   a. Compute KL(empirical || each candidate MPDAG) per group
   b. Compute α* (ideal allocation) from KL
   c. Pick group most behind target allocation (or force-explore)
   d. Draw one sample, update empirical distribution
   e. Estimate current best DAG (union of argmin KL across groups)
   f. Compute SHD against true DAG
   g. Repeat until budget exhausted
```

---

## 19. Glossary

| Term | Meaning |
|------|---------|
| DAG | Directed Acyclic Graph — causal graph with directed edges, no cycles |
| CPT | Conditional Probability Table — numbers on each node |
| MPDAG | Maximally Partially Directed Acyclic Graph — partial knowledge state |
| SHD | Structural Hamming Distance — how wrong your graph estimate is |
| KL Divergence | How different two probability distributions are |
| Intervention | Forcing a variable to a value (do-operator) |
| Causal Sufficiency | No hidden common causes |
| Latent Confounder | Hidden variable causing multiple observed variables |
| Chordal Graph | Graph where every long cycle has a shortcut edge |
| Graph Coloring | Assigning colors to nodes so adjacent nodes differ |
| α* | Ideal sample allocation fractions (from KL) |
| Golden Config | Intervention values that maximally separate candidate MPDAGs |
| Chi-Square Test | Statistical test for independence between variables |
| Track-and-Stop | Bandit algorithm adapted to causal discovery |
| Fanout | How many observed nodes each latent connects to |
| δ (delta) | Shift in probability caused by a latent confounder |
