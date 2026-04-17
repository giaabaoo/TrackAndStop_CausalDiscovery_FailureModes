"""
Robust TsP variants under unobserved confounding.

Option 1 — KL Cap: cap alpha_star[k] when min_KL[k] exceeds a threshold,
  reducing over-commitment to interventions whose distributions no candidate fits well.

Option 4 — Random Fallback: if min_KL across ALL groups stays above a threshold
  (no candidate fits well anywhere), switch to uniform random allocation for that step.

Sweep num_latents in {0,1,2,3,4,5}.
Fix n=5, density=0.5, B=5000, 50 trials, delta=0.3, fanout=2.
Baseline: standard TsP and Random included for reference.
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

from TsP import (
    shanmugam_random_chordal, convert, greedyColoring, indices_of_elements,
    adj_list_to_string_with_vertices, MPDAG,
    OrientCut_and_Enumeratmpdags, enumerate_causaleffects,
    add_latent_confounders, block_sample_intervention, Sample_and_update_dist
)
from Rnd import Learn_cut

# ── Parameters ─────────────────────────────────────────────────────────────────
NODES = 5
DEGREE = 0.5
NUM_GRAPHS = 50
B = 5000
GAP = 100
NUM_WORKERS = min(50, mp.cpu_count() - 2)
CONFOUNDER_DELTA = 0.3
FANOUT = 2
NUM_LATENTS_SWEEP = [0, 1, 2, 3, 4, 5]

# Option 1: penalize alpha inversely by min_KL — groups where no candidate fits get less weight
# alpha[k] = 1 / (min_KL[k] * min_KL[k])  instead of 1 / min_KL[k]
# This quadratically penalizes high-KL (likely confounded) groups

# Option 4: fallback to random if min_KL has not decreased by more than this fraction
# over the last WINDOW steps — stagnation = confounding signal
FALLBACK_WINDOW = 200      # steps to look back
FALLBACK_MIN_DROP = 0.05   # if min_KL drops less than 5% over window, fall back


# ── Random baseline sampler ────────────────────────────────────────────────────
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


# ── Shared setup (graph + BN + MPDAG + data) ──────────────────────────────────
def _setup(seed, nodes, degree, num_latents):
    from scipy.special import rel_entr
    np.random.seed(seed)
    random.seed(seed)

    def min_row_distance(matrix):
        from scipy.spatial.distance import cdist
        if len(matrix) < 2:
            return 0
        D = cdist(matrix, matrix)
        np.fill_diagonal(D, np.inf)
        return D.min()

    def int_to_binary_list(number, bits):
        return [int(b) for b in ("{:0" + str(bits) + "b}").format(number)]

    a = shanmugam_random_chordal(nodes, degree)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp)
    bn.generateCPTs()

    mpdg = MPDAG(bn)
    mpdg.Edges = bn.arcs()
    mpdg.Arcs = set({})

    MPDAG_LIST = []
    for i in I:
        tmp2, _ = OrientCut_and_Enumeratmpdags(mpdg, i)
        MPDAG_LIST.append(tmp2)

    Truedag = bn.arcs()
    Pv = 1
    for n in bn.nodes():
        Pv = Pv * bn.cpt(n)

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    VARS, PD, Ps, Cnts, data = [], [], [], [], []
    golden_intv = np.zeros(len(I))
    for i in I:
        V = list(mpdg.nodes - set(i))
        p_search_min = np.zeros(2 ** len(i))
        for v in range(2 ** len(i)):
            config = int_to_binary_list(v, len(i))
            tmp3, _ = enumerate_causaleffects(MPDAG_LIST[I.index(i)], V, i, Pv, config)
            tmp3 = np.array(tmp3)
            p_search_min[v] = min_row_distance(tmp3)
        golden_intv[I.index(i)] = np.argmax(p_search_min)

    for i in I:
        I_index = I.index(i)
        V = list(mpdg.nodes - set(i))
        Ps.append([0] * (2 ** len(V)))
        config = int_to_binary_list(int(golden_intv[I.index(i)]), len(i))
        tmp3, tmp1 = enumerate_causaleffects(MPDAG_LIST[I_index], V, i, Pv, config)
        PD.append(tmp3)
        Cnts.append([0] * (2 ** len(V)))
        VARS.append(tmp1)
        data.append(block_sample_intervention(bn_sample, i, tmp1, config, observed_names))

    return I, MPDAG_LIST, Truedag, Pv, Ps, PD, Cnts, VARS, data, mpdg, int_to_binary_list


# ── Standard TsP ──────────────────────────────────────────────────────────────
def run_tsp(seed, nodes, degree, Max_samples, num_latents):
    from scipy.special import rel_entr
    I, MPDAG_LIST, Truedag, Pv, Ps, PD, Cnts, VARS, data, mpdg, _ = \
        _setup(seed, nodes, degree, num_latents)

    sz_i = len(I)
    Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
    d_star = np.zeros(sz_i)
    t = 1
    samples = 0
    Dstar_E = set()

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= Max_samples:
        I_index = 0
        Dstar_E = set()
        for i in I:
            KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
            alpha_star[I_index] = 1 / max(KL_vector.min(), 1e-10)
            d_star[I_index] = np.argmin(KL_vector)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(d_star[I_index])].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


# ── Option 1: Quadratic KL Penalty ────────────────────────────────────────────
def run_tsp_klcap(seed, nodes, degree, Max_samples, num_latents):
    """alpha[k] = 1/min_KL^2 instead of 1/min_KL — quadratically penalizes
    groups where no candidate fits well (high min_KL = likely confounded)."""
    from scipy.special import rel_entr
    I, MPDAG_LIST, Truedag, Pv, Ps, PD, Cnts, VARS, data, mpdg, _ = \
        _setup(seed, nodes, degree, num_latents)

    sz_i = len(I)
    Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
    d_star = np.zeros(sz_i)
    t = 1
    samples = 0
    Dstar_E = set()

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= Max_samples:
        I_index = 0
        Dstar_E = set()
        for i in I:
            KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
            min_kl = KL_vector.min()
            # Option 1: quadratic penalty — 1/min_KL^2 instead of 1/min_KL
            alpha_star[I_index] = 1 / max(min_kl ** 2, 1e-10)
            d_star[I_index] = np.argmin(KL_vector)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(d_star[I_index])].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


# ── Option 4: Stagnation-based Random Fallback ────────────────────────────────
def run_tsp_fallback(seed, nodes, degree, Max_samples, num_latents):
    """Track min_KL over a sliding window. If it hasn't decreased by FALLBACK_MIN_DROP
    fraction over FALLBACK_WINDOW steps, the allocation has stagnated — fall back to random."""
    from scipy.special import rel_entr
    I, MPDAG_LIST, Truedag, Pv, Ps, PD, Cnts, VARS, data, mpdg, _ = \
        _setup(seed, nodes, degree, num_latents)

    sz_i = len(I)
    Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
    d_star = np.zeros(sz_i)
    t = 1
    samples = 0
    Dstar_E = set()
    global_min_kl_history = []  # track min across all groups each step

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= Max_samples:
        I_index = 0
        Dstar_E = set()
        step_min_kl = np.inf
        for i in I:
            KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
            for dindex in range(len(MPDAG_LIST[I_index])):
                kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
            min_kl = KL_vector.min()
            step_min_kl = min(step_min_kl, min_kl)
            alpha_star[I_index] = 1 / max(min_kl, 1e-10)
            d_star[I_index] = np.argmin(KL_vector)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(d_star[I_index])].arcs()
            I_index += 1

        global_min_kl_history.append(step_min_kl)

        # Option 4: stagnation detection over sliding window
        stagnated = False
        if len(global_min_kl_history) >= FALLBACK_WINDOW and min(Nt) >= 25 * math.sqrt(t):
            old_kl = global_min_kl_history[-FALLBACK_WINDOW]
            new_kl = global_min_kl_history[-1]
            if old_kl > 0 and (old_kl - new_kl) / old_kl < FALLBACK_MIN_DROP:
                stagnated = True

        if stagnated:
            act = np.random.randint(sz_i)
        else:
            s = sum(alpha_star)
            alpha_star = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
            act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


# ── Random baseline ───────────────────────────────────────────────────────────
def run_rnd_one_graph(seed, nodes, degree, Max_samples, Gap, num_latents):
    np.random.seed(seed)
    random.seed(seed)

    a = shanmugam_random_chordal(nodes, degree)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp)
    bn.generateCPTs()
    Truedag = bn.arcs()
    Edges = bn.arcs()
    arcs = set([])

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    sz_i = len(I)
    probabilities = list(np.ones(sz_i) / sz_i)
    data = [block_sample_rnd(bn_sample, i, Max_samples, observed_names) for i in I]

    index = list(np.zeros(sz_i))
    index_arr = np.zeros((Max_samples + 1, sz_i))
    for t in range(Max_samples + 1):
        acttt = np.random.choice(range(sz_i), size=1, p=probabilities)
        index[int(acttt[0])] += 1
        index_arr[t, :] = index

    for t in range(Gap, Max_samples + 1, Gap):
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data[I.index(i)],
                             int(index_arr[t, I.index(i)]), Edges)

    return len(Truedag - arcs) + len(arcs - Truedag)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Using {NUM_WORKERS} parallel workers')
    print(f'Option 1: quadratic KL penalty | Option 4: stagnation window={FALLBACK_WINDOW}, min_drop={FALLBACK_MIN_DROP}')
    results = {}

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n=== {label} ===')
        seeds = list(range(NUM_GRAPHS))

        print('  Running TsP (baseline)...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp, nodes=NODES, degree=DEGREE, Max_samples=B, num_latents=num_latents)
            tsp_shds = pool.map(fn, seeds)

        print('  Running TsP-KLCap (Option 1)...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp_klcap, nodes=NODES, degree=DEGREE, Max_samples=B, num_latents=num_latents)
            klcap_shds = pool.map(fn, seeds)

        print('  Running TsP-Fallback (Option 4)...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp_fallback, nodes=NODES, degree=DEGREE, Max_samples=B, num_latents=num_latents)
            fallback_shds = pool.map(fn, seeds)

        print('  Running Random...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_rnd_one_graph, nodes=NODES, degree=DEGREE, Max_samples=B, Gap=GAP, num_latents=num_latents)
            rnd_shds = pool.map(fn, seeds)

        results[num_latents] = {
            'tsp':      (np.mean(tsp_shds),      np.std(tsp_shds)),
            'klcap':    (np.mean(klcap_shds),    np.std(klcap_shds)),
            'fallback': (np.mean(fallback_shds), np.std(fallback_shds)),
            'rnd':      (np.mean(rnd_shds),      np.std(rnd_shds)),
        }
        print(f'  TsP:          {np.mean(tsp_shds):.2f} +/- {np.std(tsp_shds):.2f}')
        print(f'  TsP-KLCap:    {np.mean(klcap_shds):.2f} +/- {np.std(klcap_shds):.2f}')
        print(f'  TsP-Fallback: {np.mean(fallback_shds):.2f} +/- {np.std(fallback_shds):.2f}')
        print(f'  Random:       {np.mean(rnd_shds):.2f} +/- {np.std(rnd_shds):.2f}')

    # Print LaTeX tables
    print('\n\n% ── TABLE: Option 1 — KL Cap ──')
    print('\\# Latents & Random & TsP & TsP-KLCap \\\\')
    for nl in NUM_LATENTS_SWEEP:
        r = results[nl]
        label = str(nl) if nl > 0 else '0 (none)'
        print(f'{label} & ${r["rnd"][0]:.2f} \\pm {r["rnd"][1]:.2f}$ '
              f'& ${r["tsp"][0]:.2f} \\pm {r["tsp"][1]:.2f}$ '
              f'& ${r["klcap"][0]:.2f} \\pm {r["klcap"][1]:.2f}$ \\\\')

    print('\n\n% ── TABLE: Option 4 — Random Fallback ──')
    print('\\# Latents & Random & TsP & TsP-Fallback \\\\')
    for nl in NUM_LATENTS_SWEEP:
        r = results[nl]
        label = str(nl) if nl > 0 else '0 (none)'
        print(f'{label} & ${r["rnd"][0]:.2f} \\pm {r["rnd"][1]:.2f}$ '
              f'& ${r["tsp"][0]:.2f} \\pm {r["tsp"][1]:.2f}$ '
              f'& ${r["fallback"][0]:.2f} \\pm {r["fallback"][1]:.2f}$ \\\\')
