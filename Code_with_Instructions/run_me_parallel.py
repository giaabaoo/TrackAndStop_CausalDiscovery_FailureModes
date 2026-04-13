"""
Parallel reproduction of Elahi et al. baseline — TsP vs Random only.
Runs each graph independently in parallel across CPU cores.
"""
import sys
sys.path.insert(0, '.')

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

import pyAgrum as gum
import copy
import random
from scipy.special import rel_entr

from TsP import (
    shanmugam_random_chordal, convert, greedyColoring, indices_of_elements,
    adj_list_to_string_with_vertices, MPDAG,
    OrientCut_and_Enumeratmpdags, enumerate_causaleffects,
    block_sample_intervention, Sample_and_update_dist
)
from Rnd import Learn_cut

def block_sample_rnd(bn, intv, n_samples):
    bn1 = gum.BayesNet(bn)
    for j in intv:
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i, j))
        bn1.cpt(j)[:] = [0.5, 0.5]
    df = gum.generateSample(bn1, n=n_samples, name_out=None, show_progress=False,
                            with_labels=True, random_order=False)
    return df[0]

# ── Parameters ─────────────────────────────────────────────────────────────────
Max_samples = 100000
Gap = 2000
Num_Grphs = 30
NUM_WORKERS = min(50, mp.cpu_count() - 2)  # leave 2 cores free

CONFIGS = [
    # (nodes, degree, label)  — matching Elahi Fig 3 & 4
    (5,  1,    'n=5, deg=1'),
    (6,  1,    'n=6, deg=1'),
    (7,  1,    'n=7, deg=1'),
    (10, 0.10, 'n=10, deg=0.10'),
    (10, 0.15, 'n=10, deg=0.15'),
    (10, 0.20, 'n=10, deg=0.20'),
]

# ── Single-graph TsP worker ────────────────────────────────────────────────────

def run_tsp_one_graph(seed, nodes, degree, Max_samples):
    import math
    from scipy.special import rel_entr
    np.random.seed(seed)
    random.seed(seed)

    def euclidean_distance(r1, r2):
        return np.sqrt(np.sum((r1 - r2) ** 2))

    def min_row_distance(matrix):
        mind = float('inf')
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                mind = min(mind, euclidean_distance(matrix[i], matrix[j]))
        return mind

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
    shd = np.zeros(Max_samples)

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    while True:
        if samples > Max_samples:
            break
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
        if min(Nt) < 25 * math.sqrt(t):
            act = np.argmin(Nt)
        else:
            act = np.argmax(t * alpha_star - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()

        shd_now = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
        shd[samples:samples + len(I[act])] = shd_now
        t += 1
        if shd_now == 0 and t >= 1000 and len(Truedag) == len(Dstar_E):
            break
        samples += len(I[act])

    return shd.tolist()


# ── Single-graph Random worker ─────────────────────────────────────────────────

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

    index = list(np.zeros(sz_i))
    index_arr = np.zeros((Max_samples + 1, sz_i))
    data = [block_sample_rnd(bn, i, Max_samples) for i in I]

    for t in range(Max_samples + 1):
        acttt = np.random.choice(categories, size=1, p=probabilities)
        index[int(acttt[0])] += 1
        index_arr[t, :] = index

    shd = np.zeros(int(Max_samples))
    samples = 0
    for t in range(Gap, Max_samples, Gap):
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data[I.index(i)], int(index_arr[t, I.index(i)]), Edges)
        shd_now = len(Truedag - arcs) + len(arcs - Truedag)
        shd[samples:samples + math.floor(avg_int_size * Gap)] = shd_now
        samples += math.floor(avg_int_size * Gap)

    shd[0:500] = len(bn.arcs())
    return shd.tolist()


# ── Plot helper ────────────────────────────────────────────────────────────────

def plot_shd(Data_save, color_line, T, label):
    dd = np.array(Data_save)
    m = np.mean(dd, axis=0)
    sd = np.std(dd, axis=0) / math.sqrt(dd.shape[0])
    T = min(T, len(m), len(sd))
    m, sd = m[:T], sd[:T]
    mup, mlp = m + sd, m - sd
    color_area = color_line + (1 - color_line) * 2.3 / 4
    plt.plot(range(T), m, color=color_line, label=label)
    plt.fill_between(range(T), mup, mlp, color=color_area)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Using {NUM_WORKERS} parallel workers')

    # Store all results
    all_results = {}

    for nodes, degree, label in CONFIGS:
        print(f'\n=== {label} ===')
        seeds = list(range(Num_Grphs))

        print('  Running Track-and-Stop...')
        with mp.Pool(NUM_WORKERS) as pool:
            tsp_fn = partial(run_tsp_one_graph, nodes=nodes, degree=degree, Max_samples=Max_samples)
            Data_TsP = pool.map(tsp_fn, seeds)

        print('  Running Random...')
        with mp.Pool(NUM_WORKERS) as pool:
            rnd_fn = partial(run_rnd_one_graph, nodes=nodes, degree=degree, Max_samples=Max_samples, Gap=Gap)
            Data_Rnd = pool.map(rnd_fn, seeds)

        all_results[(nodes, degree)] = {
            'tsp': Data_TsP,
            'rnd': Data_Rnd,
            'label': label
        }
        # Save raw data
        np.save(f'../figures/data_tsp_{label.replace(" ", "_").replace("=","").replace(",","")}.npy', np.array(Data_TsP))
        np.save(f'../figures/data_rnd_{label.replace(" ", "_").replace("=","").replace(",","")}.npy', np.array(Data_Rnd))
        print(f'  Data saved.')

    # ── Figure 2: varying nodes (degree=1) ──
    fig2_configs = [(5, 1), (6, 1), (7, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (nodes, degree) in zip(axes, fig2_configs):
        res = all_results[(nodes, degree)]
        T = Max_samples - Gap
        dd_tsp = np.array(res['tsp'])
        dd_rnd = np.array(res['rnd'])
        for dd, color, label in [
            (dd_tsp, np.array([179, 63, 64])/255, 'Track and Stop'),
            (dd_rnd, np.array([1, 119, 179])/255, 'Random'),
        ]:
            m = np.mean(dd, axis=0)
            sd = np.std(dd, axis=0) / math.sqrt(dd.shape[0])
            T_ = min(T, len(m), len(sd))
            color_area = color + (1 - color) * 2.3 / 4
            ax.plot(range(T_), m[:T_], color=color, label=label)
            ax.fill_between(range(T_), m[:T_]+sd[:T_], m[:T_]-sd[:T_], color=color_area)
        ax.set_title(f'n={nodes}')
        ax.set_xlabel('Interventional Samples (×$10^4$)')
        ax.set_ylabel('SHD')
        ax.set_xticks([i*10000 for i in range(11)])
        ax.set_xticklabels([str(i) for i in range(11)])
        ax.grid(True)
        ax.legend()
    fig.suptitle('Figure 2: SHD vs Samples — Varying Graph Order (degree=1)', fontsize=13)
    plt.tight_layout()
    plt.savefig('../figures/figure2_varying_nodes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved: figures/figure2_varying_nodes.png')

    # ── Figure 3: varying density (n=10) ──
    fig3_configs = [(10, 0.10), (10, 0.15), (10, 0.20)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, (nodes, degree) in zip(axes, fig3_configs):
        res = all_results[(nodes, degree)]
        T = Max_samples - Gap
        dd_tsp = np.array(res['tsp'])
        dd_rnd = np.array(res['rnd'])
        for dd, color, label in [
            (dd_tsp, np.array([179, 63, 64])/255, 'Track and Stop'),
            (dd_rnd, np.array([1, 119, 179])/255, 'Random'),
        ]:
            m = np.mean(dd, axis=0)
            sd = np.std(dd, axis=0) / math.sqrt(dd.shape[0])
            T_ = min(T, len(m), len(sd))
            color_area = color + (1 - color) * 2.3 / 4
            ax.plot(range(T_), m[:T_], color=color, label=label)
            ax.fill_between(range(T_), m[:T_]+sd[:T_], m[:T_]-sd[:T_], color=color_area)
        ax.set_title(f'density={degree}')
        ax.set_xlabel('Interventional Samples (×$10^4$)')
        ax.set_ylabel('SHD')
        ax.set_xticks([i*10000 for i in range(11)])
        ax.set_xticklabels([str(i) for i in range(11)])
        ax.grid(True)
        ax.legend()
    fig.suptitle('Figure 3: SHD vs Samples — Varying Graph Density (n=10)', fontsize=13)
    plt.tight_layout()
    plt.savefig('../figures/figure3_varying_density.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: figures/figure3_varying_density.png')

    print('\nAll done!')
