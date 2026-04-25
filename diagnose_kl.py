"""Diagnose min_KL values under confounding to calibrate thresholds."""
import sys
sys.path.insert(0, './Code_with_Instructions')
sys.path.insert(0, '.')

import numpy as np
import math
import random
import pyAgrum as gum
from scipy.special import rel_entr

from TsP import (
    shanmugam_random_chordal, convert, greedyColoring, indices_of_elements,
    adj_list_to_string_with_vertices, MPDAG,
    OrientCut_and_Enumeratmpdags, enumerate_causaleffects,
    add_latent_confounders, block_sample_intervention, Sample_and_update_dist
)

NODES = 5; DEGREE = 0.5; CONFOUNDER_DELTA = 0.3; FANOUT = 2

for num_latents in [0, 1, 3]:
    min_kl_samples = []
    for seed in range(5):
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
        Pv = 1
        for n in bn.nodes(): Pv = Pv * bn.cpt(n)

        if num_latents > 0:
            bn_sample, observed_names = add_latent_confounders(bn, num_latents=num_latents, fanout=FANOUT, delta=CONFOUNDER_DELTA, seed=seed)
        else:
            observed_names = [bn.variable(n).name() for n in bn.nodes()]; bn_sample = bn
        observed_names = sorted(observed_names, key=lambda x: int(x))

        PD, Ps, Cnts, data = [], [], [], []
        golden_intv = np.zeros(len(I))
        for i in I:
            V = list(mpdg.nodes - set(i))
            p_search_min = np.zeros(2**len(i))
            for v in range(2**len(i)):
                config = int_to_binary_list(v, len(i))
                tmp3, _ = enumerate_causaleffects(MPDAG_LIST[I.index(i)], V, i, Pv, config)
                tmp3 = np.array(tmp3)
                from scipy.spatial.distance import cdist
                if len(tmp3) >= 2:
                    D = cdist(tmp3, tmp3); np.fill_diagonal(D, np.inf); p_search_min[v] = D.min()
            golden_intv[I.index(i)] = np.argmax(p_search_min)
        for i in I:
            I_index = I.index(i); V = list(mpdg.nodes - set(i))
            Ps.append([0]*(2**len(V)))
            config = int_to_binary_list(int(golden_intv[I.index(i)]), len(i))
            tmp3, tmp1 = enumerate_causaleffects(MPDAG_LIST[I_index], V, i, Pv, config)
            PD.append(tmp3); Cnts.append([0]*(2**len(V)))
            data.append(block_sample_intervention(bn_sample, i, tmp1, config, observed_names))

        sz_i = len(I); Nt = np.zeros(sz_i, dtype=int)
        for i in I:
            Sample_and_update_dist(data[I.index(i)], I.index(i), Cnts, Nt[I.index(i)])
            Nt[I.index(i)] += 1
            Ps[I.index(i)] = (np.array(Cnts[I.index(i)]) / Nt[I.index(i)]).tolist()

        # Run 500 steps and log min_KL per group
        for step in range(500):
            for I_index in range(sz_i):
                KL_vector = np.zeros(len(MPDAG_LIST[I_index]))
                for dindex in range(len(MPDAG_LIST[I_index])):
                    kl = sum(rel_entr(Ps[I_index], PD[I_index][dindex])) / math.log(2)
                    KL_vector[dindex] = kl if not math.isinf(kl) else 1e5
                min_kl_samples.append(KL_vector.min())
            act = np.argmin(Nt)
            Sample_and_update_dist(data[act], act, Cnts, Nt[act])
            Nt[act] += 1
            Ps[act] = (np.array(Cnts[act]) / Nt[act]).tolist()

    arr = np.array(min_kl_samples)
    print(f'num_latents={num_latents}: min_KL stats — mean={arr.mean():.4f}, median={np.median(arr):.4f}, p10={np.percentile(arr,10):.4f}, p90={np.percentile(arr,90):.4f}, max={arr.max():.4f}')
