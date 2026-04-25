"""
SDM Extension 4: Cumulative regret metric.

SHD at the end is a terminal metric. Cumulative regret sums SHD at every step,
penalizing both slowness and overconfidence. This gives a proper SDM evaluation.

cumulative_regret(T) = sum_{t=1}^{T} SHD(graph_estimate_at_t, true_dag)

We compute this for TsP vs Random under:
  - No confounding
  - 1 latent
  - 3 latents

Also report area-under-regret-curve (AUC) as a single scalar per trial.
Fixed: n=5, density=0.5, B=5000, 50 trials, delta=0.3, fanout=2, logged every GAP samples.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

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
GAP = 250
CONFOUNDER_DELTA = 0.3
FANOUT = 2
NUM_WORKERS = min(50, mp.cpu_count() - 2)
CHECKPOINTS = list(range(GAP, B + 1, GAP))
NUM_LATENTS_SWEEP = [0, 1, 3]


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


def run_tsp_regret(seed, num_latents):
    from scipy.special import rel_entr
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

    sz_i = len(I); Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
    t = 1; samples = 0; Dstar_E = set()

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    shd_log = []
    next_cp = 0
    cumreg = 0.0
    cumreg_curve = []

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

        shd_now = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
        cumreg += shd_now

        if next_cp < len(CHECKPOINTS) and samples >= CHECKPOINTS[next_cp]:
            cumreg_curve.append(cumreg)
            next_cp += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    while len(cumreg_curve) < len(CHECKPOINTS):
        cumreg_curve.append(cumreg)
    return cumreg_curve[:len(CHECKPOINTS)], cumreg


def run_rnd_regret(seed, num_latents):
    np.random.seed(seed); random.seed(seed)

    a = shanmugam_random_chordal(NODES, DEGREE)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    r = greedyColoring(graph_dict, len(graph_dict))
    I = indices_of_elements(r)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp); bn.generateCPTs()
    Truedag = bn.arcs(); Edges = bn.arcs()

    if num_latents > 0:
        bn_sample, observed_names = add_latent_confounders(
            bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
    else:
        observed_names = [bn.variable(n).name() for n in bn.nodes()]
        bn_sample = bn
    observed_names = sorted(observed_names, key=lambda x: int(x))

    sz_i = len(I)
    data = [block_sample_rnd(bn_sample, i, B, observed_names) for i in I]
    index = list(np.zeros(sz_i))
    index_arr = np.zeros((B + 1, sz_i))
    for t in range(B + 1):
        act = np.random.randint(sz_i)
        index[act] += 1
        index_arr[t, :] = index

    arcs = set()
    cumreg = 0.0
    cumreg_curve = []
    prev_cp = 0

    for cp in CHECKPOINTS:
        for t in range(prev_cp, cp, GAP):
            for i in I:
                arcs = Learn_cut(bn, i, arcs, data[I.index(i)],
                                 int(index_arr[min(t, B), I.index(i)]), Edges)
        shd = len(Truedag - arcs) + len(arcs - Truedag)
        # accumulate over steps between prev checkpoint and this one
        cumreg += shd * GAP
        cumreg_curve.append(cumreg)
        prev_cp = cp

    return cumreg_curve[:len(CHECKPOINTS)], cumreg


if __name__ == '__main__':
    os.makedirs('figures', exist_ok=True)
    seeds = list(range(NUM_GRAPHS))
    all_results = {}

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\nRunning {label}...')

        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_tsp_regret, num_latents=num_latents)
            tsp_raw = pool.map(fn, seeds)
        tsp_curves = np.array([r[0] for r in tsp_raw])
        tsp_aucs   = np.array([r[1] for r in tsp_raw])

        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_rnd_regret, num_latents=num_latents)
            rnd_raw = pool.map(fn, seeds)
        rnd_curves = np.array([r[0] for r in rnd_raw])
        rnd_aucs   = np.array([r[1] for r in rnd_raw])

        all_results[num_latents] = {
            'tsp_curves': tsp_curves, 'tsp_aucs': tsp_aucs,
            'rnd_curves': rnd_curves, 'rnd_aucs': rnd_aucs,
        }
        print(f'  TsP  AUC: {tsp_aucs.mean():.1f} ± {tsp_aucs.std():.1f}')
        print(f'  Rnd  AUC: {rnd_aucs.mean():.1f} ± {rnd_aucs.std():.1f}')

    # Plot cumulative regret curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    colors = ['steelblue', 'darkorange', 'green']
    for idx, num_latents in enumerate(NUM_LATENTS_SWEEP):
        ax = axes[idx]
        res = all_results[num_latents]
        xs = CHECKPOINTS

        tsp_m = res['tsp_curves'].mean(axis=0)
        tsp_s = res['tsp_curves'].std(axis=0)
        rnd_m = res['rnd_curves'].mean(axis=0)
        rnd_s = res['rnd_curves'].std(axis=0)

        ax.plot(xs, tsp_m, color=colors[idx], label='TsP', linewidth=2)
        ax.fill_between(xs, tsp_m - tsp_s, tsp_m + tsp_s, alpha=0.2, color=colors[idx])
        ax.plot(xs, rnd_m, color=colors[idx], linestyle='--', label='Random', linewidth=2)
        ax.fill_between(xs, rnd_m - rnd_s, rnd_m + rnd_s, alpha=0.15, color=colors[idx])

        title = 'No confounding' if num_latents == 0 else f'{num_latents} latent(s)'
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Interventional samples', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Cumulative SHD Regret', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cumulative Regret (Sum of SHD over Time)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/sdm4_cumulative_regret.png', dpi=150, bbox_inches='tight')
    print('\nSaved figures/sdm4_cumulative_regret.png')

    # Print AUC summary table (LaTeX)
    print('\n\n% ── TABLE: Cumulative Regret AUC ──')
    print(r'\begin{tabular}{c|cc}')
    print(r'\toprule')
    print(r'Latents & Random (AUC) & TsP (AUC) \\')
    print(r'\midrule')
    for num_latents in NUM_LATENTS_SWEEP:
        res = all_results[num_latents]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        print(f'{label} & ${res["rnd_aucs"].mean():.0f} \\pm {res["rnd_aucs"].std():.0f}$'
              f' & ${res["tsp_aucs"].mean():.0f} \\pm {res["tsp_aucs"].std():.0f}$ \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')

    np.save('figures/sdm4_data.npy', all_results)
