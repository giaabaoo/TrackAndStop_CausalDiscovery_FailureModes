"""
SDM Extension 1: SHD-vs-samples learning curves.

Plots SHD over time (every GAP samples) for TsP vs Random under:
  - No confounding (baseline)
  - 1 latent confounder
  - 3 latent confounders

Fixed: n=5, density=0.5, B=5000, 50 trials, delta=0.3, fanout=2.
Output: figures/sdm1_learning_curves.png + raw numpy arrays for LaTeX.
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
CHECKPOINTS = list(range(0, B + 1, GAP))


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


def run_tsp_curve(seed, num_latents):
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
    alpha_star = np.zeros(sz_i); d_star = np.zeros(sz_i)
    t = 1; samples = 0

    for i in I:
        Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
        Nt[I.index(i)] += 1
        Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

    shd_curve = []
    next_cp = 0

    while samples <= B:
        if next_cp < len(CHECKPOINTS) and samples >= CHECKPOINTS[next_cp]:
            Dstar_E = set()
            for idx in range(sz_i):
                KL_vector = np.zeros(len(MPDAG_LIST[idx]))
                for dindex in range(len(MPDAG_LIST[idx])):
                    kl = sum(rel_entr(Ps[idx], PD[idx][dindex])) / math.log(2)
                    KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
                Dstar_E = Dstar_E | MPDAG_LIST[idx][int(np.argmin(KL_vector))].arcs()
            shd = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
            shd_curve.append(shd)
            next_cp += 1

        I_index = 0; Dstar_E = set()
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

    while len(shd_curve) < len(CHECKPOINTS):
        shd_curve.append(shd_curve[-1] if shd_curve else len(Truedag))
    return shd_curve[:len(CHECKPOINTS)]


def run_rnd_curve(seed, num_latents):
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

    shd_curve = []
    arcs = set()
    for cp in CHECKPOINTS:
        if cp == 0:
            shd_curve.append(len(Truedag))
            continue
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data[I.index(i)],
                             int(index_arr[cp, I.index(i)]), Edges)
        shd = len(Truedag - arcs) + len(arcs - Truedag)
        shd_curve.append(shd)
    return shd_curve


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)
    seeds = list(range(NUM_GRAPHS))
    conditions = [0, 1, 3]
    colors_tsp = ['steelblue', 'darkorange', 'green']
    colors_rnd = ['steelblue', 'darkorange', 'green']
    labels_tsp = ['TsP (0 latents)', 'TsP (1 latent)', 'TsP (3 latents)']
    labels_rnd = ['Random (0 latents)', 'Random (1 latent)', 'Random (3 latents)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    all_results = {}

    for idx, num_latents in enumerate(conditions):
        print(f'\nRunning num_latents={num_latents}...')
        fn_tsp = partial(run_tsp_curve, num_latents=num_latents)
        fn_rnd = partial(run_rnd_curve, num_latents=num_latents)

        with mp.Pool(NUM_WORKERS) as pool:
            tsp_curves = np.array(pool.map(fn_tsp, seeds))
        with mp.Pool(NUM_WORKERS) as pool:
            rnd_curves = np.array(pool.map(fn_rnd, seeds))

        all_results[num_latents] = {'tsp': tsp_curves, 'rnd': rnd_curves}

        tsp_mean = tsp_curves.mean(axis=0)
        tsp_std  = tsp_curves.std(axis=0)
        rnd_mean = rnd_curves.mean(axis=0)
        rnd_std  = rnd_curves.std(axis=0)
        xs = CHECKPOINTS

        ax = axes[idx]
        ax.plot(xs, tsp_mean, color=colors_tsp[idx], label='TsP', linewidth=2)
        ax.fill_between(xs, tsp_mean - tsp_std, tsp_mean + tsp_std,
                        alpha=0.2, color=colors_tsp[idx])
        ax.plot(xs, rnd_mean, color=colors_rnd[idx], linestyle='--', label='Random', linewidth=2)
        ax.fill_between(xs, rnd_mean - rnd_std, rnd_mean + rnd_std,
                        alpha=0.15, color=colors_rnd[idx])
        title = 'No confounding' if num_latents == 0 else f'{num_latents} latent(s)'
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Interventional samples', fontsize=11)
        if idx == 0:
            ax.set_ylabel('SHD', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        print(f'  TsP final: {tsp_mean[-1]:.2f} ± {tsp_std[-1]:.2f}')
        print(f'  Rnd final: {rnd_mean[-1]:.2f} ± {rnd_std[-1]:.2f}')

    plt.suptitle('SHD vs. Interventional Samples under Confounding', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('figures/sdm1_learning_curves.png', dpi=150, bbox_inches='tight')
    print('\nSaved figures/sdm1_learning_curves.png')

    np.save('figures/sdm1_data.npy', all_results)

    # Print summary table
    print('\n\nSummary at each checkpoint:')
    print(f'{"Samples":>8}', end='')
    for nl in conditions:
        print(f'  TsP({nl}L)  Rnd({nl}L)', end='')
    print()
    for ci, cp in enumerate(CHECKPOINTS):
        print(f'{cp:>8}', end='')
        for nl in conditions:
            tm = all_results[nl]['tsp'][:, ci].mean()
            rm = all_results[nl]['rnd'][:, ci].mean()
            print(f'  {tm:6.2f}    {rm:6.2f}', end='')
        print()
