import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')

import TsP
import Rnd

Max_samples = 100000
Gap = 2000
Num_Grphs = 50
nodes = 5
degree = 1

def plot_shd(Data_save, color_line, plt_obj, T, plt_label):
    dd = np.array(Data_save)
    m = np.mean(dd, axis=0)
    sd = np.std(dd, axis=0) / math.sqrt(dd.shape[0])
    T = min(T, len(m))
    m = m[0:T]
    mup = (m + sd[0:T])
    mlp = (m - sd[0:T])
    color_area = color_line + (1 - color_line) * 2.3 / 4
    plt_obj.plot(range(T), m, color=color_line, label=plt_label)
    plt_obj.fill_between(range(T), mup, mlp, color=color_area)
    plt_obj.xlabel('Interventional Samples')
    plt_obj.ylabel('SHD')
    plt_obj.grid(True)
    plt_obj.legend()
    return plt_obj

print('Running Track and Stop...')
Data_TsP = TsP.Track_and_stop(Num_Grphs, nodes, degree, Max_samples)
color_line = np.array([179, 63, 64]) / 255
plt = plot_shd(Data_TsP, color_line, plt, Max_samples - Gap, 'Track and Stop')

print('Running Random...')
Data_Rnd = Rnd.Random_Interventions(Num_Grphs, nodes, degree, Max_samples, Gap)
color_line = np.array([1, 119, 179]) / 255
plt = plot_shd(Data_Rnd, color_line, plt, Max_samples - Gap, 'Random')

plt.title(f'SHD vs Samples (n={nodes}, degree={degree}) — Quick Reproduction')
plt.savefig(f'../baseline_quick_n{nodes}_deg{degree}.png', dpi=150, bbox_inches='tight')
print(f'Plot saved to baseline_quick_n{nodes}_deg{degree}.png')
