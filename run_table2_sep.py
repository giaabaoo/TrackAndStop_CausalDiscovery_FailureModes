"""
Table 2 (Separating Sets variant): Compare three policies under unobserved confounding.
  - TsP-Color:  original graph-coloring intervention groups (our run_table2.py)
  - TsP-Sep:    strongly separating sets (Harsh's TsP_multinode.py idea, from NeurIPS 2017)
  - Random:     uniform random baseline

Sweep num_latents in {0,1,2,3,4,5}.
Fix n=5, density=0.5, B=5000, 50 trials, delta=0.3, fanout=2.
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


# ── Strongly separating sets (from NeurIPS 2017 / Harsh's TsP_multinode.py) ──
def generate_strongly_separating_sets(n_nodes):
    m = math.ceil(math.log2(n_nodes)) if n_nodes > 1 else 1
    sets = []
    for bit in range(m):
        S = [node for node in range(n_nodes) if ((node >> bit) & 1)]
        if S:
            sets.append(S)
    return sets


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


# ── Shared TsP core — takes I (intervention groups) as argument ───────────────
def _run_tsp_core(seed, nodes, degree, Max_samples, num_latents, use_sep_sets):
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
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp)
    bn.generateCPTs()

    # Choose intervention groups
    if use_sep_sets:
        I = generate_strongly_separating_sets(nodes)
    else:
        r = greedyColoring(graph_dict, len(graph_dict))
        I = indices_of_elements(r)

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


def run_tsp_color(seed, nodes, degree, Max_samples, num_latents):
    return _run_tsp_core(seed, nodes, degree, Max_samples, num_latents, use_sep_sets=False)

def run_tsp_sep(seed, nodes, degree, Max_samples, num_latents):
    return _run_tsp_core(seed, nodes, degree, Max_samples, num_latents, use_sep_sets=True)


# ── Random single graph ────────────────────────────────────────────────────────
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
    categories = range(sz_i)
    probabilities = list(np.ones(sz_i) / sz_i)

    data = [block_sample_rnd(bn_sample, i, Max_samples, observed_names) for i in I]

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

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n=== {label} ===')
        seeds = list(range(NUM_GRAPHS))

        print('  Running TsP-Color...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp_color, nodes=NODES, degree=DEGREE, Max_samples=B, num_latents=num_latents)
            color_shds = pool.map(fn, seeds)

        print('  Running TsP-Sep...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp_sep, nodes=NODES, degree=DEGREE, Max_samples=B, num_latents=num_latents)
            sep_shds = pool.map(fn, seeds)

        print('  Running Random...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_rnd_one_graph, nodes=NODES, degree=DEGREE, Max_samples=B, Gap=GAP, num_latents=num_latents)
            rnd_shds = pool.map(fn, seeds)

        results[num_latents] = {
            'color_mean': np.mean(color_shds), 'color_std': np.std(color_shds),
            'sep_mean':   np.mean(sep_shds),   'sep_std':   np.std(sep_shds),
            'rnd_mean':   np.mean(rnd_shds),   'rnd_std':   np.std(rnd_shds),
        }
        print(f'  TsP:       {np.mean(color_shds):.2f} +/- {np.std(color_shds):.2f}')
        print(f'  TsP-Sep:   {np.mean(sep_shds):.2f} +/- {np.std(sep_shds):.2f}')
        print(f'  Random:    {np.mean(rnd_shds):.2f} +/- {np.std(rnd_shds):.2f}')

    # Print LaTeX
    print('\n\n% ── TABLE: Separating Sets vs Graph Coloring vs Random ──')
    print('Latents & Random & TsP & TsP-Sep \\\\')
    for num_latents in NUM_LATENTS_SWEEP:
        r = results[num_latents]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        print(f'{label} & ${r["rnd_mean"]:.2f} \\pm {r["rnd_std"]:.2f}$ '
              f'& ${r["color_mean"]:.2f} \\pm {r["color_std"]:.2f}$ '  # TsP
              f'& ${r["sep_mean"]:.2f} \\pm {r["sep_std"]:.2f}$ \\\\')
