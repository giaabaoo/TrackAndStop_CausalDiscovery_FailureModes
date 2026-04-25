"""
SDM Extension 2: UCB-style exploration bonus for intervention selection.

Standard TsP allocation: act = argmax(t * alpha - Nt)
UCB variant:             act = argmax(t * alpha - Nt + beta * sqrt(log(t) / Nt))

Under confounding, KL estimates mislead alpha. The UCB bonus forces continued
exploration of undersampled interventions rather than locking onto a wrong hypothesis.

Compare: TsP (standard) vs TsP-UCB (beta=0.5, 1.0, 2.0) vs Random
Sweep num_latents in {0, 1, 3}.
Fixed: n=5, density=0.5, B=5000, 50 trials, delta=0.3, fanout=2.
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

NODES = 5
DEGREE = 0.5
NUM_GRAPHS = 50
B = 5000
GAP = 100
CONFOUNDER_DELTA = 0.3
FANOUT = 2
NUM_WORKERS = min(50, mp.cpu_count() - 2)
UCB_BETAS = [0.0, 0.5, 1.0, 2.0]  # 0.0 = standard TsP
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


def run_tsp_ucb(seed, num_latents, beta):
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
            d_star[I_index] = np.argmin(KL_vector)
            Dstar_E = Dstar_E | MPDAG_LIST[I_index][int(d_star[I_index])].arcs()
            I_index += 1

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i

        # UCB bonus on top of allocation score
        if min(Nt) < 25 * math.sqrt(t):
            act = np.argmin(Nt)
        else:
            ucb_bonus = beta * np.sqrt(np.log(t) / np.maximum(Nt, 1))
            act = np.argmax(t * alpha_star_norm - Nt + ucb_bonus)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    return len(Truedag - Dstar_E) + len(Dstar_E - Truedag)


def run_rnd_one(seed, num_latents):
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
    for t in range(GAP, B + 1, GAP):
        for i in I:
            arcs = Learn_cut(bn, i, arcs, data[I.index(i)],
                             int(index_arr[t, I.index(i)]), Edges)
    return len(Truedag - arcs) + len(arcs - Truedag)


if __name__ == '__main__':
    seeds = list(range(NUM_GRAPHS))
    results = {}

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n=== {label} ===')
        results[num_latents] = {}

        for beta in UCB_BETAS:
            name = f'TsP' if beta == 0.0 else f'TsP-UCB(β={beta})'
            print(f'  Running {name}...')
            with mp.Pool(NUM_WORKERS) as pool:
                fn = partial(run_tsp_ucb, num_latents=num_latents, beta=beta)
                shds = pool.map(fn, seeds)
            results[num_latents][beta] = {'mean': np.mean(shds), 'std': np.std(shds)}
            print(f'    SHD: {np.mean(shds):.2f} ± {np.std(shds):.2f}')

        print('  Running Random...')
        with mp.Pool(NUM_WORKERS) as pool:
            fn = partial(run_rnd_one, num_latents=num_latents)
            rnd_shds = pool.map(fn, seeds)
        results[num_latents]['rnd'] = {'mean': np.mean(rnd_shds), 'std': np.std(rnd_shds)}
        print(f'    Random SHD: {np.mean(rnd_shds):.2f} ± {np.std(rnd_shds):.2f}')

    # LaTeX table
    print('\n\n% ── TABLE: UCB Exploration Bonus ──')
    header = 'Latents & Random & TsP'
    for beta in UCB_BETAS[1:]:
        header += f' & TsP-UCB($\\beta={beta}$)'
    print(header + ' \\\\')
    for num_latents in NUM_LATENTS_SWEEP:
        r = results[num_latents]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        row = f'{label} & ${r["rnd"]["mean"]:.2f} \\pm {r["rnd"]["std"]:.2f}$'
        row += f' & ${r[0.0]["mean"]:.2f} \\pm {r[0.0]["std"]:.2f}$'
        for beta in UCB_BETAS[1:]:
            row += f' & ${r[beta]["mean"]:.2f} \\pm {r[beta]["std"]:.2f}$'
        print(row + ' \\\\')
