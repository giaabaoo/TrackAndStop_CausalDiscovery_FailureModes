"""
Oracle Intervention Design Experiment.

Hypothesis: The failure under confounding is partly due to *which* nodes are
intervened on. Standard TsP uses graph-coloring groups designed for the observed
DAG. Under confounding, the latent variable H creates an "extra path" that these
groups don't block.

Oracle idea: what if we intervene on nodes that are *most likely* to be children
of latents — i.e., we prioritize variables with high unexplained marginal variance?

This tests the following mathematically grounded idea:
  - Under no confounding, P(Y | do(X)) = sum_pa P(Y | X, pa) P(pa)
  - Under confounding, P(Y | do(X)) != P(Y | do(X)) computed from MPDAG
    because latent H creates backdoor paths not blocked by do(X).
  - Nodes with high marginal entropy H(Y) relative to their structural children
    are more likely to be confounded.

We compare:
  TsP-Entropy: prioritize intervening on high-entropy nodes first.
    Allocation: same as TsP, but the graph-coloring is replaced by entropy-ranked
    singleton interventions. Within the Track-and-Stop framework, this changes I.

Also: TsP-Variance — within each graph-coloring group, pick the intervention
configuration that maximizes empirical variance of P_emp across checkpoints.
This is a proper exploration bonus grounded in active learning (maximizing
information gain approximated by output variance).
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


def marginal_entropy(data_obs, node_name, n_obs=2000):
    """Empirical H(X) from first n_obs rows."""
    col = data_obs[node_name][:n_obs].values.astype(float)
    p1 = col.mean()
    p0 = 1 - p1
    if p0 <= 0 or p1 <= 0:
        return 0.0
    return -p0 * math.log2(p0) - p1 * math.log2(p1)


def run_tsp_entropy_ranked(seed, num_latents):
    """
    TsP with entropy-ranked singleton intervention groups.
    Nodes with high H(X) are intervened on first (more likely confounded).
    Intervention groups: singletons sorted by descending marginal entropy.
    """
    np.random.seed(seed); random.seed(seed)

    def int_to_binary_list(number, bits):
        return [int(b) for b in ("{:0" + str(bits) + "b}").format(number)]

    a = shanmugam_random_chordal(NODES, DEGREE)
    adjacency_list = list(a.edges)
    graph_dict = convert(adjacency_list)
    tmp = adj_list_to_string_with_vertices(adjacency_list)
    bn = gum.fastBN(tmp); bn.generateCPTs()

    mpdg = MPDAG(bn); mpdg.Edges = bn.arcs(); mpdg.Arcs = set()

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

    # Estimate marginal entropy from observational data
    obs_df, _ = gum.generateSample(bn_sample, n=2000, name_out=None,
                                   show_progress=False, with_labels=True, random_order=False)
    obs_df = obs_df[observed_names]
    node_ids = list(mpdg.nodes)
    node_names = [bn.variable(n).name() for n in node_ids]
    entropies = {nid: marginal_entropy(obs_df, nm) for nid, nm in zip(node_ids, node_names)
                 if nm in observed_names}

    # Use entropy-ranked singleton intervention groups
    sorted_nodes = sorted(entropies.keys(), key=lambda n: entropies[n], reverse=True)
    I = [[n] for n in sorted_nodes]  # singletons, high-entropy first

    MPDAG_LIST = []
    for i in I:
        tmp2, _ = OrientCut_and_Enumeratmpdags(mpdg, i)
        MPDAG_LIST.append(tmp2)

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

    sz_i = len(I); Nt = np.zeros(sz_i, dtype=int)
    alpha_star = np.zeros(sz_i)
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
    Truedag = bn.arcs(); Edges = bn.arcs()
    sz_i = len(I)

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

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n{"="*60}\n  {label}\n{"="*60}')

        for name, fn_raw in [
            ('TsP (baseline)',     run_tsp_baseline),
            ('TsP-EntropyRanked',  run_tsp_entropy_ranked),
            ('Random',             run_rnd),
        ]:
            fn = partial(fn_raw, num_latents=num_latents)
            with mp.Pool(NUM_WORKERS) as pool:
                shds = pool.map(fn, seeds)
            m, s = np.mean(shds), np.std(shds)
            print(f'  {name:22s}  SHD: {m:.2f} ± {s:.2f}')
