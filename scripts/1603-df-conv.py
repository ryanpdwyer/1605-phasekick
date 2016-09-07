from __future__ import division
from os import path
from tqdm import tqdm
import numpy as np
import phasekick
import phasekickstan
import time
from collections import OrderedDict
import lockin
import h5py
import datetime
import pystan

print("tr-EFM Convolution model")
print("====================================")
print("\n")

def arrays_to_stan_data(t, mu_df, sigma_df, N):
    df_std = sigma_df / N**0.5
    t = t * 1e3 # s to ms
    m = t <= 0
    y_neg = mu_df[m]
    offset = np.mean(y_neg)
    return {
    'y_neg': y_neg - offset,
    't': t[~m], # only positive times
    'N_neg': t[m].size,
    'y_neg_err': df_std[m],
    'y': mu_df[~m] - offset,
    'y_err': df_std[~m],
    'offset': offset,
    'N': t[~m].size
    }

def get_filename_without_ext(path_):
    return path.splitext(path.split(path_)[1])[0]

def get_t_mu_sigma_df(trefm):
    t = trefm.tm
    mu_df = trefm.dfm
    sigma_df = np.std(trefm.df, axis=0, ddof=1)
    return t, mu_df, sigma_df

def save_workup(gr, params, t, mu_df, sigma_df, intensity, N):
    gr['t'] = t
    gr['mu_df'] = mu_df
    gr['sigma_df'] = sigma_df
    gr['N'] = N
    gr.create_group('workup_params')
    gr.create_group('experiment_params')
    for key, val in params.items():  # Save workup params
        gr['workup_params'][key] = val
    gr['experiment_params']['intensity'] = intensity


fname = '2015-12-17/151217-234238-20sun-df-384.h5'

intens = 20.
params = {'fp': 4000,
         'fc': 15000,
         't_phase': -0.052,
         'ti': -500e-6,
         'tf': 2.8e-3,
         'Ndec': 1}


with tqdm(total=384, leave=True, unit='B', unit_scale=True) as pbar:
    with h5py.File(fname, 'r') as fh:
        trefm = phasekick.AverageTrEFM.from_group(fh['data'],
                                              params['fp'],
                                              params['fc'],
                                              params['t_phase'],
                                              params['ti'],
                                              params['tf'], pbar=pbar)

        t, mu_df, sigma_df = get_t_mu_sigma_df(trefm)
        M, N = trefm.t.shape


fs = 1e6
dt = 1e-6
fir = lockin.lock2(62000, params['fp'], params['fc'], fs)
K = fir.size
NK1 = N + K - 1
K2 = int((K-1)/2)
t_offset = -dt*K2
tdelay = -4e-6
t_eval = np.arange(NK1)*dt + t[0] + t_offset + tdelay


offset = np.mean(mu_df[t < 0.])


data = {
    't_eval': t_eval*1e3, # ms,
    'N': N,
    'K': K,
    'kern': fir,
    'y_err': sigma_df,
    'y': mu_df - offset,
     'mu_df0': 0,
     'mu_df_inf': -20,
     'mu_tau': np.array([0.5,2.5]),
     'sigma_df0': 2.5,
     'sigma_df_inf': 7,
}

pm = phasekickstan.PhasekickModel('dflive_doub_conv', data, priors={})

pm.run(chains=4, iter=1000)

pm.save('1603-df-conv.h5')
