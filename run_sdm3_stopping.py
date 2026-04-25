"""
SDM Extension 3: Stopping rule analysis.

Track-and-Stop has an implicit stopping criterion: stop when the current best
hypothesis is confident enough (min KL across intervention groups crosses a threshold).
Under confounding, the algorithm may stop too early (false confidence) or drift
without converging.

We measure:
  - Time-to-stop (number of samples when min_KL first exceeds threshold TAU)
  - False stop rate (stopped but final SHD > 0)
  - Never-stop rate (min_KL never crosses TAU within budget B)

Sweep num_latents in {0, 1, 2, 3}, TAU in {0.5, 1.0, 2.0}.
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

NODES = 5
DEGREE = 0.5
NUM_GRAPHS = 50
B = 5000
CONFOUNDER_DELTA = 0.3
FANOUT = 2
NUM_WORKERS = min(50, mp.cpu_count() - 2)
TAUS = [0.5, 1.0, 2.0]
NUM_LATENTS_SWEEP = [0, 1, 2, 3]
NEVER_STOP_SENTINEL = B + 1


def run_stopping_analysis(seed, num_latents, tau):
    """
    Returns (time_to_stop, is_false_stop, final_shd).
    time_to_stop = NEVER_STOP_SENTINEL if never triggered.
    """
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

    time_to_stop = NEVER_STOP_SENTINEL
    stopped = False

    while samples <= B:
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

        # Check stopping condition
        if not stopped and global_min_kl >= tau:
            time_to_stop = samples
            stopped = True

        s = sum(alpha_star)
        alpha_star_norm = alpha_star / s if s > 0 else np.ones(sz_i) / sz_i
        act = np.argmin(Nt) if min(Nt) < 25 * math.sqrt(t) else np.argmax(t * alpha_star_norm - Nt)

        Sample_and_update_dist(data[act], act, Cnts, Nt[act])
        Nt[act] += 1
        Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()
        t += 1
        samples += len(I[act])

    final_shd = len(Truedag - Dstar_E) + len(Dstar_E - Truedag)
    is_false_stop = stopped and (final_shd > 0)
    return time_to_stop, int(is_false_stop), final_shd, int(stopped)


if __name__ == '__main__':
    seeds = list(range(NUM_GRAPHS))
    results = {}

    for num_latents in NUM_LATENTS_SWEEP:
        label = f'{num_latents} latent(s)' if num_latents > 0 else 'No confounding'
        print(f'\n=== {label} ===')
        results[num_latents] = {}

        for tau in TAUS:
            print(f'  tau={tau}...')
            with mp.Pool(NUM_WORKERS) as pool:
                fn = partial(run_stopping_analysis, num_latents=num_latents, tau=tau)
                raw = pool.map(fn, seeds)

            times = np.array([r[0] for r in raw])
            false_stops = np.array([r[1] for r in raw])
            final_shds = np.array([r[2] for r in raw])
            stopped = np.array([r[3] for r in raw])

            never_stop_rate = (times == NEVER_STOP_SENTINEL).mean()
            stop_times = times[times < NEVER_STOP_SENTINEL]
            mean_time = stop_times.mean() if len(stop_times) > 0 else float('nan')
            false_stop_rate = false_stops[stopped == 1].mean() if stopped.sum() > 0 else 0.0

            results[num_latents][tau] = {
                'never_stop_rate': never_stop_rate,
                'mean_stop_time': mean_time,
                'false_stop_rate': false_stop_rate,
                'mean_final_shd': final_shds.mean(),
            }
            print(f'    Never-stop rate: {never_stop_rate:.2f}')
            print(f'    Mean stop time (among stopped): {mean_time:.1f}')
            print(f'    False-stop rate (stopped but SHD>0): {false_stop_rate:.2f}')
            print(f'    Mean final SHD: {final_shds.mean():.2f}')

    # LaTeX table: one per tau
    for tau in TAUS:
        print(f'\n\n% ── TABLE: Stopping Analysis (tau={tau}) ──')
        print(r'\begin{tabular}{c|cccc}')
        print(r'\toprule')
        print(r'Latents & Never-stop & Mean stop time & False-stop rate & Mean SHD \\')
        print(r'\midrule')
        for num_latents in NUM_LATENTS_SWEEP:
            r = results[num_latents][tau]
            label = str(num_latents) if num_latents > 0 else '0 (none)'
            ns = r['never_stop_rate']
            mt = r['mean_stop_time']
            fs = r['false_stop_rate']
            ms = r['mean_final_shd']
            mt_str = f'{mt:.0f}' if not math.isnan(mt) else 'N/A'
            print(f'{label} & {ns:.2f} & {mt_str} & {fs:.2f} & {ms:.2f} \\\\')
        print(r'\bottomrule')
        print(r'\end{tabular}')

    # Compact single summary table (tau=1.0)
    print('\n\n% ── COMPACT TABLE: Stopping Rule (tau=1.0) ──')
    tau = 1.0
    print(r'\begin{tabular}{c|ccc}')
    print(r'\toprule')
    print(r'Latents & Never-stop (\%) & False-stop (\%) & Mean SHD \\')
    print(r'\midrule')
    for num_latents in NUM_LATENTS_SWEEP:
        r = results[num_latents][tau]
        label = str(num_latents) if num_latents > 0 else '0 (none)'
        print(f'{label} & {100*r["never_stop_rate"]:.0f} & {100*r["false_stop_rate"]:.0f} & {r["mean_final_shd"]:.2f} \\\\')
    print(r'\bottomrule')
    print(r'\end{tabular}')
