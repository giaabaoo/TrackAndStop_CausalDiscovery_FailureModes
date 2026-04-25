"""
Confounder Screening Experiment.

Hypothesis: Before running TsP, use a small observational sample to test for
confounding via the faithfulness assumption violation. Under causal sufficiency,
all conditional independences in the data should be explained by d-separation.
Under confounding, there exist pairs (X, Y) that appear dependent despite being
d-separated in the observed graph — the "signature" of hidden common causes.

Algorithm:
  1. Draw N_screen = 500 observational samples.
  2. For each edge (X, Y) in the MPDAG, run a chi-square test of
     X _|_ Y | Pa(X) ∩ Pa(Y) in the observational data.
  3. If any test rejects H0 (independence) at alpha=0.05, flag as confounded.
  4. If flagged: use Random allocation + Learn_cut.
     If not flagged: use standard TsP.

Mathematical grounding:
  - Under causal sufficiency + faithfulness, the PC algorithm's independence
    tests all hold in the population. Violation detects confounding.
  - This is exactly the FCI (Fast Causal Inference) algorithm's first step,
    applied as a pre-filter rather than a full ADMG search.
  - The screen uses O(N_screen * |E|) tests, a small overhead vs. B=5000.

Compare TsP-Screen vs TsP vs Random across num_latents in {0,1,2,3}.
"""
import sys
sys.path.insert(0, './Code_with_Instructions')
sys.path.insert(0, '.')

import numpy as np
import math
import random
import multiprocessing as mp
from functools import partial
import pyAgrum as gum
from scipy.special import rel_entr
from scipy.stats import chi2_contingency

from TsP import (
    shanmugam_random_chordal, convert, greedyColoring, indices_of_elements,
    adj_list_to_string_with_vertices, MPDAG,
    OrientCut_and_Enumeratmpdags, enumerate_causaleffects,
    add_latent_confounders, block_sample_intervention, Sample_and_update_dist
)
from Rnd import Learn_cut

NODES = 5
DEGREE = 0.5
NUM_GRAPHS = 50
B = 5000
GAP = 100
CONFOUNDER_DELTA = 0.3
FANOUT = 2
NUM_WORKERS = min(50, mp.cpu_count() - 2)
NUM_LATENTS_SWEEP = [0, 1, 2, 3]
N_SCREEN = 3000         # observational samples for confounding screen
SCREEN_ALPHA = 0.10     # significance level for chi-square test (liberal to boost recall)


def block_sample_rnd(bn_env, intv, n_samples, observed_names):
    bn1 = gum.BayesNet(bn_env)
    for j in intv:
        for parent_id in list(bn1.parents(j)):
            if not bn1.variable(parent_id).name().startswith("H"):
                bn1.eraseArc(gum.Arc(parent_id, j))
        shape_without_child = bn1.cpt(j).toarray().shape[:-1]
        uniform = np.full(shape_without_child + (2,), 0.5, dtype=float)
        bn1.cpt(j).fillWith(uniform.flatten().tolist())
    result = gum.generateSample(bn1, n=n_samples, name_out=None,
                                show_progress=False, with_labels=True,
                                random_order=False)
    obs_sorted = sorted(observed_names, key=lambda x: int(x))
    return result[0][obs_sorted]


def detect_confounding(bn_sample, observed_names, bn_clean, n_screen=N_SCREEN, alpha=SCREEN_ALPHA):
    """
    Detect confounding via conditional independence violations.

    Under causal sufficiency + faithfulness, for every node X:
        X _|_ Y | Pa(X)  for all Y not in {X} union Pa(X) union descendants(X)
    Latent confounders create spurious dependencies that violate these tests.

    We test X _|_ Y | Pa(X) for non-descendant, non-parent Y.
    If any test rejects, we flag confounding.
    """
    obs_df, _ = gum.generateSample(bn_sample, n=n_screen, name_out=None,
                                   show_progress=False, with_labels=True, random_order=False)
    obs_df = obs_df[observed_names]

    node_ids = sorted(bn_clean.nodes())
    id_to_name = {n: bn_clean.variable(n).name() for n in node_ids}
    arcs = bn_clean.arcs()  # set of (parent, child) tuples

    for child in node_ids:
        child_name = id_to_name[child]
        if child_name not in obs_df.columns:
            continue
        parents = [p for (p, c) in arcs if c == child]
        parent_names = [id_to_name[p] for p in parents
                        if id_to_name[p] in obs_df.columns]

        # Find non-parents, non-children to test independence given Pa(child)
        children_of_child = [c for (p, c) in arcs if p == child]
        exclude = set(parents + children_of_child + [child])
        others = [n for n in node_ids if n not in exclude]

        for other in others:
            other_name = id_to_name[other]
            if other_name not in obs_df.columns:
                continue

            if not parent_names:
                # No parents: test marginal independence child _|_ other
                ct = np.zeros((2, 2), dtype=int)
                for cv in [0, 1]:
                    for ov in [0, 1]:
                        ct[cv, ov] = ((obs_df[child_name] == cv) &
                                      (obs_df[other_name] == ov)).sum()
                if ct.sum() < 30:
                    continue
                try:
                    _, p_val, _, _ = chi2_contingency(ct)
                    if p_val < alpha:
                        return True
                except Exception:
                    continue
            else:
                # Test child _|_ other | parents via Mantel-Haenszel-style stratification
                # For each parent configuration, build a 2x2 table and combine p-values
                p_vals = []
                for pv in range(2 ** len(parent_names)):
                    bits = [int(b) for b in f'{pv:0{len(parent_names)}b}']
                    mask = np.ones(len(obs_df), dtype=bool)
                    for pname, pval in zip(parent_names, bits):
                        mask &= (obs_df[pname] == pval).values
                    sub = obs_df[mask]
                    if len(sub) < 20:
                        continue
                    ct = np.zeros((2, 2), dtype=int)
                    for cv in [0, 1]:
                        for ov in [0, 1]:
                            ct[cv, ov] = ((sub[child_name] == cv) &
                                          (sub[other_name] == ov)).sum()
                    if ct.min() < 2:
                        continue
                    try:
                        _, p_val, _, _ = chi2_contingency(ct)
                        p_vals.append(p_val)
                    except Exception:
                        continue

                # If any stratum shows significant dependence, flag confounding
                if p_vals and min(p_vals) < alpha:
                    return True
    return False


def run_tsp_screen(seed, num_latents):
    """TsP with confounding pre-screen. Falls back to Random if confounded."""
    np.random.seed(seed); random.seed(seed)

    def int_to_binary_list(number, bits):
        return [int(b) for b in ("{:0" + str(bits) + "b}").format(number)]

    a = shanmugam_random_chordal(NODES, DEGREE)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp); bn.generateCPTs()

    mpdg = MPDAG(bn); mpdg.Edges = bn.arcs(); mpdg.Arcs = set()
    MPDAG_LIST = []
    for i in I:
        tmp2, _ = OrientCut_and_Enumeratmpdags(mpdg, i)
        MPDAG_LIST.append(tmp2)

    Truedag = bn.arcs()
    Pv = 1
    for n in bn.nodes(): Pv = Pv * bn.cpt(n)

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    # ── Confounding screen ────────────────────────────────────────────────────
    is_confounded = detect_confounding(bn_sample, observed_names, bn)

    data_rnd = [block_sample_rnd(bn_sample, i, B, observed_names) for i in I]
    sz_i = len(I)

    if is_confounded:
        # Fall back to Random + Learn_cut
        index = list(np.zeros(sz_i))
        index_arr = np.zeros((B + 2, sz_i))
        for t in range(B + 1):
            act = np.random.randint(sz_i)
            index[act] += 1
            index_arr[t, :] = index
        arcs = set()
        for t in range(GAP, B + 1, GAP):
            for i in I:
                arcs = Learn_cut(bn, i, arcs, data_rnd[I.index(i)],
                                 int(index_arr[t, I.index(i)]), bn.arcs())
        return len(Truedag - arcs) + len(arcs - Truedag)

    # ── Standard TsP ─────────────────────────────────────────────────────────
    PD, Ps, Cnts, data = [], [], [], []
    golden_intv = np.zeros(len(I))
    for i in I:
        V = list(mpdg.nodes - set(i))
        p_search_min = np.zeros(2 ** len(i))
        for v in range(2 ** len(i)):
            config = int_to_binary_list(v, len(i))
            tmp3, _ = enumerate_causaleffects(MPDAG_LIST[I.index(i)], V, i, Pv, config)
            tmp3 = np.array(tmp3)
            from scipy.spatial.distance import cdist
            if len(tmp3) >= 2:
                D = cdist(tmp3, tmp3); np.fill_diagonal(D, np.inf)
                p_search_min[v] = D.min()
        golden_intv[I.index(i)] = np.argmax(p_search_min)

    for i in I:
        I_index = I.index(i); V = list(mpdg.nodes - set(i))
        Ps.append([0] * (2 ** len(V)))
        config = int_to_binary_list(int(golden_intv[I.index(i)]), len(i))
        tmp3, tmp1 = enumerate_causaleffects(MPDAG_LIST[I_index], V, i, Pv, config)
        PD.append(tmp3); Cnts.append([0] * (2 ** len(V)))
        data.append(block_sample_intervention(bn_sample, i, tmp1, config, observed_names))

    Nt = np.zeros(sz_i, dtype=int); alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= B:
        I_index = 0; Dstar_E = set()
        for i in I:
            KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
            alpha_star[I_index] = 1 / max(KL_vector.min(), 1e-10)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(np.argmin(KL_vector))].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)
        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1; samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


def run_tsp_baseline(seed, num_latents):
    np.random.seed(seed); random.seed(seed)

    def int_to_binary_list(number, bits):
        return [int(b) for b in ("{:0" + str(bits) + "b}").format(number)]

    a = shanmugam_random_chordal(NODES, DEGREE)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp); bn.generateCPTs()
    mpdg = MPDAG(bn); mpdg.Edges = bn.arcs(); mpdg.Arcs = set()
    MPDAG_LIST = []
    for i in I:
        tmp2, _ = OrientCut_and_Enumeratmpdags(mpdg, i)
        MPDAG_LIST.append(tmp2)
    Truedag = bn.arcs()
    Pv = 1
    for n in bn.nodes(): Pv = Pv * bn.cpt(n)

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    PD, Ps, Cnts, data = [], [], [], []
    golden_intv = np.zeros(len(I))
    for i in I:
        V = list(mpdg.nodes - set(i))
        p_search_min = np.zeros(2 ** len(i))
        for v in range(2 ** len(i)):
            config = int_to_binary_list(v, len(i))
            tmp3, _ = enumerate_causaleffects(MPDAG_LIST[I.index(i)], V, i, Pv, config)
            tmp3 = np.array(tmp3)
            from scipy.spatial.distance import cdist
            if len(tmp3) >= 2:
                D = cdist(tmp3, tmp3); np.fill_diagonal(D, np.inf)
                p_search_min[v] = D.min()
        golden_intv[I.index(i)] = np.argmax(p_search_min)

    for i in I:
        I_index = I.index(i); V = list(mpdg.nodes - set(i))
        Ps.append([0] * (2 ** len(V)))
        config = int_to_binary_list(int(golden_intv[I.index(i)]), len(i))
        tmp3, tmp1 = enumerate_causaleffects(MPDAG_LIST[I_index], V, i, Pv, config)
        PD.append(tmp3); Cnts.append([0] * (2 ** len(V)))
        data.append(block_sample_intervention(bn_sample, i, tmp1, config, observed_names))

    sz_i = len(I); Nt = np.zeros(sz_i, dtype=int); alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= B:
        I_index = 0; Dstar_E = set()
        for i in I:
            KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
            alpha_star[I_index] = 1 / max(KL_vector.min(), 1e-10)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(np.argmin(KL_vector))].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)
        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1; samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


def run_rnd(seed, num_latents):
    np.random.seed(seed); random.seed(seed)

    a = shanmugam_random_chordal(NODES, DEGREE)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp); bn.generateCPTs()
    Truedag = bn.arcs(); Edges = bn.arcs(); sz_i = len(I)

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    data_rnd = [block_sample_rnd(bn_sample, i, B, observed_names) for i in I]
    index = list(np.zeros(sz_i))
    index_arr = np.zeros((B + 2, sz_i))
    for t in range(B + 1):
        act = np.random.randint(sz_i)
        index[act] += 1
        index_arr[t, :] = index

    arcs = set()
    for t in range(GAP, B + 1, GAP):
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data_rnd[I.index(i)],
                             int(index_arr[t, I.index(i)]), Edges)
    return len(Truedag - arcs) + len(arcs - Truedag)


if __name__ == '__main__':
    seeds = list(range(NUM_GRAPHS))

    # First: measure screen accuracy
    print('=== Confounding Screen Accuracy ===')
    for num_latents in NUM_LATENTS_SWEEP:
        detections = []
        for seed in seeds:
            np.random.seed(seed); random.seed(seed)
            a = shanmugam_random_chordal(NODES, DEGREE)
            adjacency_list = list(a.edges)
            tmp = adj_list_to_string_with_vertices(adjacency_list)
            bn = gum.fastBN(tmp); bn.generateCPTs()
            if num_latents > 0:
                bn_sample, observed_names = add_latent_confounders(
                    bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
            else:
                observed_names = [bn.variable(n).name() for n in bn.nodes()]
                bn_sample = bn
            observed_names = sorted(observed_names, key=lambda x: int(x))
            detected = detect_confounding(bn_sample, observed_names, bn)
            detections.append(int(detected))
        rate = np.mean(detections)
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'  {label}: detection rate = {rate:.2f} ({"true positive" if num_latents>0 else "false positive"})')

    print('\n=== SHD Results ===')
    results = {}
    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n  --- {label} ---')
        results[num_latents] = {}

        for name, fn_raw in [
            ('TsP (baseline)', run_tsp_baseline),
            ('TsP-Screen',     run_tsp_screen),
            ('Random',         run_rnd),
        ]:
            fn = partial(fn_raw, num_latents=num_latents)
            with mp.Pool(NUM_WORKERS) as pool:
                shds = pool.map(fn, seeds)
            m, s = np.mean(shds), np.std(shds)
            results[num_latents][name] = (m, s)
            print(f'    {name:20s}  SHD: {m:.2f} ± {s:.2f}')

    print('\n\n% ── TABLE: Confounder-Screened TsP ──')
    print(r'\begin{tabular}{c|ccc}')
    print(r'\toprule')
    print(r'Latents & Random & TsP & TsP-Screen \\')
    print(r'\midrule')
    for num_latents in NUM_LATENTS_SWEEP:
        r = results[num_latents]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        rnd_m, rnd_s = r['Random']
        tsp_m, tsp_s = r['TsP (baseline)']
        sc_m, sc_s = r['TsP-Screen']
        print(f'{label} & ${rnd_m:.2f} \\pm {rnd_s:.2f}$ & ${tsp_m:.2f} \\pm {tsp_s:.2f}$ & ${sc_m:.2f} \\pm {sc_s:.2f}$ \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
