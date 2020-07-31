import sys
sys.path.append('../code')
from plotting import *
from analysis import *
import pickle
import os
from glob import glob
import re
import numpy as np

timecourse_path = 'timecourses'
plot_path = '../plots'
#fake appropriate params
fake_params = ({'time_vector': np.arange(0,30000,1)},)

if __name__ == '__main__':
    files = glob(os.path.join(timecourse_path, '*.pic'))
    r = re.compile('[0-9]\.[0-9]+')
    params = []
    for f in files:
        params.append([float(x) for x in r.findall(f)])
    params = np.array(params)
    idx = params[:,1].argsort()
    params = params[idx, :]
    files = [files[x] for x in idx]
    dominance_ratios = np.empty(params.shape, dtype=np.float)
    mean_durations = np.empty(params.shape, dtype=np.float)
    timecourses = []
    for i, (f, p) in enumerate(zip(files, params)):
        try:
            with open(f, 'rb') as _f:
                timecourse = pickle.load(_f)
                timecourses.append(timecourse)
            _ratio = ratio_dominance(timecourse, 1)
            durations_1, durations_2 = dominance_durations(timecourse,fake_params)
            dominance_ratios[i] = (_ratio, 1 - _ratio)
            mean_durations[i] = (durations_1.mean(), durations_2.mean())
        except:
            print('Complete dominance')
            dominance_ratios[i] = (1,0) if _ratio > .5 else (0,1)
            mean_durations[i] = np.zeros(2)
            mean_durations[i,dominance_ratios[i].argmax()] = np.inf
    
    fig, axs = plot_cmp_dominance(
        x_ticks=params[:,1], 
        duration_1=mean_durations[:,0], 
        duration_2=mean_durations[:,1],
        dominance=dominance_ratios[:,0],
        xlabel='Half difference in attention weights',
        title='Influence of differential attention weights'
    )
    figname = 'dominance_cmp.png'
    savefig(fig, figname, plot_path)
    
    figs = []
    for p, tc in zip(params, timecourses):
        fig, axs = plot_timecourse(tc)
        name = f'timecourse_weight-attention_mean-{p[0]}_diff-{p[1]}.png'
        savefig(fig, name, plot_path)
        figs.append(fig)
        