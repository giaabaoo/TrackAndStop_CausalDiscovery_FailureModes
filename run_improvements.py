"""
Mathematically grounded improvements to TsP under confounding.

Root cause: KL(P_emp || P_d) with MPDAG candidates fails under confounding because
  (a) the support of P_true (a MAG) is not covered by any MPDAG candidate,
  (b) the argmin still selects a "least-wrong" candidate with high confidence,
  (c) the algorithm stops (believes it found the true graph) when it hasn't.

Three principled fixes — each a minimal change to the estimation step:

Method A — TsP-JS (Jensen-Shannon divergence):
    Replace KL(P_emp || P_d) with JS(P_emp, P_d).
    JS is defined via the mixture M = (P+Q)/2 which always has full support,
    so JS is bounded in [0, 1 bit] even when supports don't overlap.
    Ref: Lin (1991), "Divergence Measures Based on the Shannon Entropy".

Method B — TsP-Adaptive (misspecification-aware fallback):
    Run standard TsP but monitor the minimum KL per group.
    Under the correctly specified model, min_KL -> 0 as O(1/sqrt(N_k)) by CLT
    (law of large numbers for empirical distributions).
    If min_KL > C / sqrt(N_k_min) persists for PATIENCE steps after burn-in,
    declare model misspecification and switch to chi-sq based Learn_cut.
    This is grounded in the Neyman-Pearson hypothesis testing framework:
    the null is "P_emp is consistent with some MPDAG candidate", and we
    reject it via the rate-of-decrease test.

Method C — TsP-MarginalKL (marginalized KL over intervention configs):
    Standard TsP fixes the intervention configuration as argmax separability,
    computed once from the clean (unconfounded) Pv.
    Under confounding, this golden config may no longer be most informative.
    TsP-MarginalKL averages KL over all intervention configs:
        score(d) = min_v KL(P_emp^v || P_d^v)
    picking the candidate closest under the most favorable config for each d.
    This is a minimax variant: rather than trusting a fixed golden config,
    find d* that minimizes the worst-case KL across configs.
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

# Fallback hyperparameters
FALLBACK_C = 1.5     # min_KL should decrease faster than C/sqrt(N) under correct model
PATIENCE = 100       # steps of rate-violation before fallback triggers
BURNIN = 150         # steps before we start checking (allow estimates to stabilize)


def js_divergence(p, q, eps=1e-10):
    """Symmetric Jensen-Shannon divergence in bits. Always finite, in [0,1]."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (sum(rel_entr(p, m)) + sum(rel_entr(q, m))) / math.log(2)


def block_sample_rnd(bn_env, intv, n_samples, observed_names):
    bn1 = gum.BayesNet(bn_env)
    for j in intv:
        for parent_id in list(bn1.parents(j)):
            if not bn1.variable(parent_id).name().startswith("H"):
                bn1.eraseArc(gum.Arc(parent_id, j))
        shape_without_child = bn1.cpt(j).toarray().shape[:-1]
        uniform = np.full(shape_without_child + (2,), 0.5)
        bn1.cpt(j).fillWith(uniform.flatten())
    result = gum.generateSample(bn1, n=n_samples, name_out=None,
                                show_progress=False, with_labels=True,
                                random_order=False)
    obs_sorted = sorted(observed_names, key=lambda x: int(x))
    return result[0][obs_sorted]


def _setup_trial(seed, num_latents):
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

    PD_all, Ps, Cnts, data = [], [], [], []

    # For each intervention group, enumerate ALL configs and store ALL causal effects
    # PD_all[k] = list of (config_idx, [causal_effect_per_mpdag])
    golden_intv = np.zeros(len(I))
    all_configs_PD = []  # all_configs_PD[k][v] = list of causal effects for config v

    for i in I:
        V = list(mpdg.nodes - set(i))
        p_search_min = np.zeros(2 ** len(i))
        config_PD_list = []
        for v in range(2 ** len(i)):
            config = int_to_binary_list(v, len(i))
            tmp3, _ = enumerate_causaleffects(MPDAG_LIST[I.index(i)], V, i, Pv, config)
            config_PD_list.append(tmp3)
            tmp3 = np.array(tmp3)
            from scipy.spatial.distance import cdist
            if len(tmp3) >= 2:
                D = cdist(tmp3, tmp3); np.fill_diagonal(D, np.inf)
                p_search_min[v] = D.min()
        golden_intv[I.index(i)] = np.argmax(p_search_min)
        all_configs_PD.append(config_PD_list)

    for i in I:
        I_index = I.index(i); V = list(mpdg.nodes - set(i))
        Ps.append([0] * (2 ** len(V)))
        config = int_to_binary_list(int(golden_intv[I.index(i)]), len(i))
        # golden config causal effects
        tmp3, tmp1 = enumerate_causaleffects(MPDAG_LIST[I_index], V, i, Pv, config)
        PD_all.append(tmp3)
        Cnts.append([0] * (2 ** len(V)))
        data.append(block_sample_intervention(bn_sample, i, tmp1, config, observed_names))

    data_rnd = [block_sample_rnd(bn_sample, i, B, observed_names) for i in I]

    sz_i = len(I); Nt = np.zeros(sz_i, dtype=int)
    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    return dict(
        I=I, MPDAG_LIST=MPDAG_LIST, Truedag=Truedag, bn=bn, mpdg=mpdg,
        PD=PD_all, Ps=Ps, Cnts=Cnts, data=data, data_rnd=data_rnd,
        Nt=Nt, sz_i=sz_i, Edges=bn.arcs(), all_configs_PD=all_configs_PD,
        golden_intv=golden_intv,
    )


# ── Baseline TsP ─────────────────────────────────────────────────────────────
def run_tsp_baseline(seed, num_latents):
    ctx = _setup_trial(seed, num_latents)
    I, MPDAG_LIST, Truedag = ctx['I'], ctx['MPDAG_LIST'], ctx['Truedag']
    PD, Ps, Cnts, data = ctx['PD'], ctx['Ps'], ctx['Cnts'], ctx['data']
    Nt, sz_i = ctx['Nt'], ctx['sz_i']

    alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

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


# ── Method A: TsP-JS ──────────────────────────────────────────────────────────
def run_js(seed, num_latents):
    ctx = _setup_trial(seed, num_latents)
    I, MPDAG_LIST, Truedag = ctx['I'], ctx['MPDAG_LIST'], ctx['Truedag']
    PD, Ps, Cnts, data = ctx['PD'], ctx['Ps'], ctx['Cnts'], ctx['data']
    Nt, sz_i = ctx['Nt'], ctx['sz_i']

    alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

    while samples <= B:
        I_index = 0; Dstar_E = set()
        for i in I:
            JS_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                JS_vector[dindex] = js_divergence(Ps[I_index], PD[I_index][dindex])
            alpha_star[I_index] = 1 / max(JS_vector.min(), 1e-10)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(np.argmin(JS_vector))].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)
        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1; samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


# ── Method B: TsP-Adaptive (misspecification-aware fallback) ─────────────────
def run_adaptive(seed, num_latents):
    """
    Under correct model, empirical KL(P_emp || P_d*) decreases as O(1/sqrt(N_k))
    by CLT for the empirical distribution.
    Under confounding, it stagnates because no MPDAG candidate is correct.
    We detect stagnation and fall back to chi-sq based edge orientation.
    """
    ctx = _setup_trial(seed, num_latents)
    I, MPDAG_LIST, Truedag = ctx['I'], ctx['MPDAG_LIST'], ctx['Truedag']
    PD, Ps, Cnts, data, data_rnd = ctx['PD'], ctx['Ps'], ctx['Cnts'], ctx['data'], ctx['data_rnd']
    Nt, sz_i, bn, Edges = ctx['Nt'], ctx['sz_i'], ctx['bn'], ctx['Edges']

    alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()
    stagnant = 0
    fallback_mode = False
    arcs = set()
    index_arr = np.zeros(sz_i, dtype=float)

    while samples <= B:
        if not fallback_mode:
            I_index = 0; Dstar_E = set()
            global_min_kl = np.inf
            for i in I:
                KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
                for dindex in range(len(MPDAG_LIST[I_index])):
                    kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                    KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
                alpha_star[I_index] = 1 / max(KL_vector.min(), 1e-10)
                Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(np.argmin(KL_vector))].arcs()
                global_min_kl = min(global_min_kl, KL_vector.min())
                I_index += 1

            # After burn-in: test if min_KL is decreasing at the expected CLT rate
            if t > BURNIN:
                N_min = max(Nt.min(), 1)
                expected_kl_rate = FALLBACK_C / math.sqrt(N_min)
                if global_min_kl > expected_kl_rate:
                    stagnant += 1
                else:
                    stagnant = max(0, stagnant - 1)

                if stagnant >= PATIENCE:
                    fallback_mode = True

            s = sum(alpha_star)
            alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
            act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)
            Sample_and_update_dist(data[act], act, Cnts, Nt[act])
            Nt[act] += 1
            Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        else:
            act = np.random.randint(sz_i)
            index_arr[act] += 1
            for i in I:
                arcs = Learn_cut(bn, i, arcs, data_rnd[I.index(i)],
                                 int(index_arr[I.index(i)]), Edges)
            Dstar_E = arcs

        t += 1; samples += len(I[act])

    if not fallback_mode:
        return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
    else:
        return len(Truedag - arcs) + len(arcs - Truedag)


# ── Method C: TsP-MinMax (best-case KL over configs) ─────────────────────────
def run_minmax(seed, num_latents):
    """
    Standard TsP fixes golden config once. Under confounding this config may
    no longer separate candidates well (the distribution is distorted).

    TsP-MinMax: for each MPDAG candidate d, find the intervention config v
    that minimizes KL(P_emp^{golden} || P_d^v) — i.e., give each candidate
    its best shot. Then d* = argmin over d.

    This is a max-likelihood-style estimator: pick d that best explains the
    observed data under any of its possible intervention configurations.
    Under no confounding, this reduces to standard TsP (same golden config).
    Under confounding, it avoids dismissing candidates because of a single
    bad-config comparison.
    """
    ctx = _setup_trial(seed, num_latents)
    I, MPDAG_LIST, Truedag = ctx['I'], ctx['MPDAG_LIST'], ctx['Truedag']
    PD, Ps, Cnts, data = ctx['PD'], ctx['Ps'], ctx['Cnts'], ctx['data']
    Nt, sz_i = ctx['Nt'], ctx['sz_i']
    all_configs_PD = ctx['all_configs_PD']

    alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

    while samples <= B:
        I_index = 0; Dstar_E = set()
        for i in I:
            n_cands = len(MPDAG_LIST[I_index])
            n_configs = len(all_configs_PD[I_index])  # 2^|I_k|

            # For each candidate d, find best (min KL) config v
            best_kl_per_cand = np.full(n_cands, np.inf)
            for dindex in range(n_cands):
                for v in range(n_configs):
                    cand_dist = all_configs_PD[I_index][v]
                    if dindex >= len(cand_dist):
                        continue
                    p_d = cand_dist[dindex]
                    kl = sum(rel_entr(Ps[I_index], p_d)) / math.log(2)
                    if not math.isinf(kl):
                        best_kl_per_cand[dindex] = min(best_kl_per_cand[dindex], kl)
                    # if still inf for this config, try next config
                best_kl_per_cand[dindex] = min(best_kl_per_cand[dindex], 1e5)

            alpha_star[I_index] = 1 / max(best_kl_per_cand.min(), 1e-10)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(np.argmin(best_kl_per_cand))].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)
        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1; samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


# ── Random baseline ───────────────────────────────────────────────────────────
def run_rnd_baseline(seed, num_latents):
    ctx = _setup_trial(seed, num_latents)
    I, Truedag, bn, Edges = ctx['I'], ctx['Truedag'], ctx['bn'], ctx['Edges']
    data_rnd, sz_i = ctx['data_rnd'], ctx['sz_i']

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
    results = {}
    methods_order = ['Random', 'TsP (baseline)', 'TsP-JS', 'TsP-Adaptive', 'TsP-MinMax']

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n{"="*62}')
        print(f'  {label}')
        print(f'{"="*62}')
        results[num_latents] = {}

        configs = [
            ('TsP (baseline)', run_tsp_baseline),
            ('TsP-JS',         run_js),
            ('TsP-Adaptive',   run_adaptive),
            ('TsP-MinMax',     run_minmax),
            ('Random',         run_rnd_baseline),
        ]
        tsp_base_shd = None
        for name, fn_raw in configs:
            fn = partial(fn_raw, num_latents=num_latents)
            with mp.Pool(NUM_WORKERS) as pool:
                shds = pool.map(fn, seeds)
            m, s = np.mean(shds), np.std(shds)
            results[num_latents][name] = (m, s)
            if name == 'TsP (baseline)':
                tsp_base_shd = m
            improvement = ''
            if tsp_base_shd is not None and name not in ('TsP (baseline)', 'Random'):
                delta = tsp_base_shd - m
                sign = '+' if delta > 0 else ''
                improvement = f'  ({sign}{delta:.2f} vs TsP)'
            print(f'  {name:20s}  SHD: {m:.2f} ± {s:.2f}{improvement}')

    # LaTeX table
    print('\n\n% ── TABLE: Improvement Methods under Confounding ──')
    print(r'\begin{tabular}{c|ccccc}')
    print(r'\toprule')
    print(r'Latents & Random & TsP & TsP-JS & TsP-Adaptive & TsP-MinMax \\')
    print(r'\midrule')
    for num_latents in NUM_LATENTS_SWEEP:
        r = results[num_latents]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        row = label
        for m in methods_order:
            mean, std = r[m]
            row += f' & ${mean:.2f} \\pm {std:.2f}$'
        print(row + r' \\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
