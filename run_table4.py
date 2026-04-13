"""
Table 4: Effect of intervention budget B.
Fix n=5, density=0.5, 50 trials.
Vary B in {1000, 2500, 5000}.
Report final SHD mean +/- std for TsP and Random.
"""
import sys
sys.path.insert(0, './Code_with_Instructions')

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
    block_sample_intervention, Sample_and_update_dist
)
from Code_with_Instructions.Rnd import Learn_cut

# ── Parameters ─────────────────────────────────────────────────────────────────
NODES = 5
DEGREE = 0.5
NUM_GRAPHS = 50
BUDGETS = [1000, 2500, 5000]
GAP = 100  # resolution for random baseline
NUM_WORKERS = min(50, mp.cpu_count() - 2)


# ── TsP single graph ───────────────────────────────────────────────────────────
def run_tsp_one_graph(seed, nodes, degree, Max_samples):
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
    Pv = 1
    for n in bn.nodes():
        Pv = Pv * bn.cpt(n)

    mpdg = MPDAG(bn)
    mpdg.Edges = bn.arcs()
    mpdg.Arcs = set({})

    MPDAG_LIST = []
    for i in I:
        tmp2, _ = OrientCut_and_Enumeratmpdags(mpdg, i)
        MPDAG_LIST.append(tmp2)

    Truedag = bn.arcs()
    bn.generateCPTs()
    Pv = 1
    for n in bn.nodes():
        Pv = Pv * bn.cpt(n)

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
        data.append(block_sample_intervention(bn, i, tmp1, config))

    sz_i = len(I)
    Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
    d_star = np.zeros(sz_i)
    t = 1
    samples = 0
    Dstar_E = set({})

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while samples <= Max_samples:
        I_index = 0
        Dstar_E = set({})
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


# ── Random single graph ────────────────────────────────────────────────────────
def block_sample_rnd(bn, intv, n_samples):
    bn1 = gum.BayesNet(bn)
    for j in intv:
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i, j))
        bn1.cpt(j)[:] = [0.5, 0.5]
    df = gum.generateSample(bn1, n=n_samples, name_out=None, show_progress=False,
                            with_labels=True, random_order=False)
    return df[0]


def run_rnd_one_graph(seed, nodes, degree, Max_samples, Gap):
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
    bn.generateCPTs()
    arcs = set([])

    sz_i = len(I)
    avg_int_size = sum(len(i) for i in I) / sz_i
    categories = range(sz_i)
    probabilities = list(np.ones(sz_i) / sz_i)

    # pre-sample enough data
    data = [block_sample_rnd(bn, i, Max_samples) for i in I]

    index = list(np.zeros(sz_i))
    index_arr = np.zeros((Max_samples + 1, sz_i))
    for t in range(Max_samples + 1):
        acttt = np.random.choice(categories, size=1, p=probabilities)
        index[int(acttt[0])] += 1
        index_arr[t, :] = index

    for t in range(Gap, Max_samples + 1, Gap):
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data[I.index(i)],
                             int(index_arr[t, I.index(i)]), Edges)

    return len(Truedag - arcs) + len(arcs - Truedag)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f'Using {NUM_WORKERS} parallel workers')
    results = {}

    for B in BUDGETS:
        print(f'\n=== Budget B={B} ===')
        seeds = list(range(NUM_GRAPHS))

        print('  Running TsP...')
        with mp.Pool(NUM_WORKERS) as pool:
            tsp_fn = partial(run_tsp_one_graph, nodes=NODES, degree=DEGREE, Max_samples=B)
            tsp_shds = pool.map(tsp_fn, seeds)

        print('  Running Random...')
        with mp.Pool(NUM_WORKERS) as pool:
            rnd_fn = partial(run_rnd_one_graph, nodes=NODES, degree=DEGREE,
                             Max_samples=B, Gap=max(GAP, B // 20))
            rnd_shds = pool.map(rnd_fn, seeds)

        results[B] = {
            'tsp_mean': np.mean(tsp_shds), 'tsp_std': np.std(tsp_shds),
            'rnd_mean': np.mean(rnd_shds), 'rnd_std': np.std(rnd_shds),
        }
        print(f'  TsP: {np.mean(tsp_shds):.2f} +/- {np.std(tsp_shds):.2f}')
        print(f'  Rnd: {np.mean(rnd_shds):.2f} +/- {np.std(rnd_shds):.2f}')

    # Print LaTeX
    print('\n\n% ── TABLE 4 LaTeX ──')
    for B in BUDGETS:
        r = results[B]
        print(f'{B} & ${r["rnd_mean"]:.2f} \\pm {r["rnd_std"]:.2f}$ & ${r["tsp_mean"]:.2f} \\pm {r["tsp_std"]:.2f}$ \\\\')
