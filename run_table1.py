"""
Experiment runner for Table 1a and Table 1b.

Table 1a: Fix density=0.5, vary delta in {0.05, 0.10, 0.20, 0.30, 0.40}
Table 1b: Fix delta=0.40, vary density in {0.1, 0.3, 0.5, 0.7, 0.9}

Both report mean +/- std SHD over 50 trials at budget B=5000.
"""

import sys
sys.path.insert(0, './Code_with_Instructions')

import numpy as np
import pyAgrum as gum
import networkx as nx
import random
import math
from scipy.special import rel_entr
import copy

# ── shared helpers (duplicated from TsP/Rnd to keep this file self-contained) ──

from TsP import (
    shanmugam_random_chordal, convert, greedyColoring, indices_of_elements,
    adj_list_to_string_with_vertices, MPDAG,
    OrientCut_and_Enumeratmpdags, enumerate_causaleffects,
    block_sample_intervention, Sample_and_update_dist
)
from Rnd import Learn_cut


def generate_cpts_with_delta(bn, delta):
    """Set each CPT row to [0.5 - delta, 0.5 + delta] or flipped, randomly."""
    for node in bn.nodes():
        cpt = bn.cpt(node)
        arr = cpt.toarray()
        # arr shape: (2,) if no parents, (2^n_parents, 2) otherwise
        flat = arr.reshape(-1, 2)
        for row in range(flat.shape[0]):
            sign = np.random.choice([-1, 1])
            p = 0.5 + sign * delta
            p = np.clip(p, 0.01, 0.99)
            flat[row] = [1 - p, p]
        cpt[:] = flat.reshape(arr.shape)


def run_track_and_stop(Num_Grphs, nodes, degree, Max_samples, delta):
    """Track-and-Stop with delta-controlled CPTs. Returns list of final SHDs."""
    def euclidean_distance(r1, r2):
        return np.sqrt(np.sum((r1 - r2) ** 2))

    def min_row_distance(matrix):
        mind = float('inf')
        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                mind = min(mind, euclidean_distance(matrix[i], matrix[j]))
        return mind

    def int_to_binary_list(number, bits):
        fmt = "{:0" + str(bits) + "b}"
        return [int(b) for b in fmt.format(number)]

    final_shds = []

    for grphs_no in range(0, Num_Grphs, 5):
        a = shanmugam_random_chordal(nodes, degree)
        adjacency_list = list(a.edges)
        graph_dict = convert(adjacency_list)
        r = greedyColoring(graph_dict, len(graph_dict))
        I = indices_of_elements(r)

        tmp = adj_list_to_string_with_vertices(adjacency_list)
        bn = gum.fastBN(tmp)
        generate_cpts_with_delta(bn, delta)

        Pv = 1
        for n in bn.nodes():
            Pv = Pv * bn.cpt(n)

        mpdg = MPDAG(bn)
        mpdg.Edges = bn.arcs()
        mpdg.Arcs = set({})

        I = indices_of_elements(r)
        MPDAG_LIST = []
        Cuts = []
        for i in I:
            tmp2, Cts = OrientCut_and_Enumeratmpdags(mpdg, i)
            MPDAG_LIST.append(tmp2)
            Cuts.append(Cts)

        for iii in range(5):
            if grphs_no + iii >= Num_Grphs:
                break
            print(f'  [TsP] Graph {grphs_no + iii}')

            Truedag = bn.arcs()
            generate_cpts_with_delta(bn, delta)
            Pv = 1
            for n in bn.nodes():
                Pv = Pv * bn.cpt(n)

            mpdag = copy.deepcopy(mpdg)

            VARS, PD, Ps, Cnts, data = [], [], [], [], []

            # golden intervention config
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

            for i in I:
                Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
                Nt[I.index(i)] += 1
                Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

            final_shd = len(Truedag)  # worst case
            while True:
                if samples > Max_samples:
                    break
                I_index = 0
                Dstar_E = set({})
                for i in I:
                    KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
                    for dindex, m in enumerate(MPDAG_LIST[I_index]):
                        kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                        KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
                    alpha_star[I_index] = 1 / KL_vector.min()
                    d_star[I_index] = np.argmin(KL_vector)
                    Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(d_star[I_index])].arcs()
                    I_index += 1

                alpha_star = alpha_star / sum(alpha_star)
                if min(Nt) < 25 * math.sqrt(t):
                    act = np.argmin(Nt)
                else:
                    act = np.argmax(t * alpha_star - Nt)

                Sample_and_update_dist(data[act], act, Cnts, Nt[act])
                Nt[act] += 1
                Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()

                shd_now = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
                t += 1
                if shd_now == 0 and t >= 1000 and len(Truedag) == len(Dstar_E):
                    final_shd = 0
                    break
                samples += len(I[act])

            final_shd = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
            final_shds.append(final_shd)

    return final_shds


def run_random(Num_Grphs, nodes, degree, Max_samples, Gap, delta):
    """Random interventions with delta-controlled CPTs. Returns list of final SHDs."""
    final_shds = []
    for grphs_no in range(Num_Grphs):
        print(f'  [Rnd] Graph {grphs_no}')
        a = shanmugam_random_chordal(nodes, degree)
        adjacency_list = list(a.edges)
        graph_dict = convert(adjacency_list)
        r = greedyColoring(graph_dict, len(graph_dict))
        I = indices_of_elements(r)

        tmp = adj_list_to_string_with_vertices(adjacency_list)
        bn = gum.fastBN(tmp)
        generate_cpts_with_delta(bn, delta)

        Truedag = bn.arcs()
        Edges = bn.arcs()

        generate_cpts_with_delta(bn, delta)
        arcs = set([])
        sz_i = len(I)
        categories = range(sz_i)
        probabilities = list(np.ones(sz_i) / sz_i)

        avg_int_size = sum(len(i) for i in I) / sz_i

        index = list(np.zeros(sz_i))
        index_arr = np.zeros((Max_samples + 1, sz_i))
        data = []
        for i in I:
            data.append(block_sample_intervention_rnd(bn, i, Max_samples))

        for t in range(Max_samples + 1):
            acttt = np.random.choice(categories, size=1, p=probabilities)
            index[int(acttt[0])] += 1
            index_arr[t, :] = index

        for t in range(Gap, Max_samples, Gap):
            for i in I:
                arcs = Learn_cut(bn, i, arcs, data[I.index(i)], int(index_arr[t, I.index(i)]), Edges)

        final_shd = len(Truedag - arcs) + len(arcs - Truedag)
        final_shds.append(final_shd)

    return final_shds


def block_sample_intervention_rnd(bn, intv, n_samples):
    bn1 = gum.BayesNet(bn)
    for j in intv:
        for i in bn1.parents(j):
            bn1.eraseArc(gum.Arc(i, j))
        bn1.cpt(j)[:] = [0.5, 0.5]
    df = gum.generateSample(bn1, n=n_samples, name_out=None, show_progress=False,
                            with_labels=True, random_order=False)
    return df[0]


# ── Experiment parameters ──────────────────────────────────────────────────────

NUM_GRAPHS = 50
NODES = 5
MAX_SAMPLES = 5000
GAP = 500

# Table 1a: vary delta, fix density=0.5
DELTAS = [0.05, 0.10, 0.20, 0.30, 0.40]
FIXED_DENSITY = 0.5

# Table 1b: vary density, fix delta=0.40
DENSITIES = [0.1, 0.3, 0.5, 0.7, 0.9]
FIXED_DELTA = 0.40

# ── Run Table 1a ───────────────────────────────────────────────────────────────

print("=" * 60)
print("TABLE 1a: vary delta, density=0.5")
print("=" * 60)

results_1a = {}
for delta in DELTAS:
    print(f"\ndelta={delta}")
    tsp = run_track_and_stop(NUM_GRAPHS, NODES, FIXED_DENSITY, MAX_SAMPLES, delta)
    rnd = run_random(NUM_GRAPHS, NODES, FIXED_DENSITY, MAX_SAMPLES, GAP, delta)
    results_1a[delta] = {
        'tsp_mean': np.mean(tsp), 'tsp_std': np.std(tsp),
        'rnd_mean': np.mean(rnd), 'rnd_std': np.std(rnd),
    }
    print(f"  TsP: {np.mean(tsp):.2f} ± {np.std(tsp):.2f}")
    print(f"  Rnd: {np.mean(rnd):.2f} ± {np.std(rnd):.2f}")

# ── Run Table 1b ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TABLE 1b: vary density, delta=0.40")
print("=" * 60)

results_1b = {}
for density in DENSITIES:
    print(f"\ndensity={density}")
    tsp = run_track_and_stop(NUM_GRAPHS, NODES, density, MAX_SAMPLES, FIXED_DELTA)
    rnd = run_random(NUM_GRAPHS, NODES, density, MAX_SAMPLES, GAP, FIXED_DELTA)
    results_1b[density] = {
        'tsp_mean': np.mean(tsp), 'tsp_std': np.std(tsp),
        'rnd_mean': np.mean(rnd), 'rnd_std': np.std(rnd),
    }
    print(f"  TsP: {np.mean(tsp):.2f} ± {np.std(tsp):.2f}")
    print(f"  Rnd: {np.mean(rnd):.2f} ± {np.std(rnd):.2f}")

# ── Print LaTeX ────────────────────────────────────────────────────────────────

print("\n\n% ── TABLE 1a LaTeX ──")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Mean SHD (mean $\pm$ std, lower is better) under varying signal strength $\delta$. Fixed: $n=5$, density$=0.5$, $B=5000$, 50 trials.}")
print(r"\begin{tabular}{c|cc}")
print(r"\toprule")
print(r"$\delta$ & Random & Track-and-Stop \\")
print(r"\midrule")
for delta in DELTAS:
    r = results_1a[delta]
    print(f"{delta:.2f} & {r['rnd_mean']:.2f} $\\pm$ {r['rnd_std']:.2f} & {r['tsp_mean']:.2f} $\\pm$ {r['tsp_std']:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")

print("\n% ── TABLE 1b LaTeX ──")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\caption{Mean SHD (mean $\pm$ std, lower is better) under varying graph density. Fixed: $n=5$, $\delta=0.40$, $B=5000$, 50 trials.}")
print(r"\begin{tabular}{c|cc}")
print(r"\toprule")
print(r"Density & Random & Track-and-Stop \\")
print(r"\midrule")
for density in DENSITIES:
    r = results_1b[density]
    print(f"{density:.1f} & {r['rnd_mean']:.2f} $\\pm$ {r['rnd_std']:.2f} & {r['tsp_mean']:.2f} $\\pm$ {r['tsp_std']:.2f} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
