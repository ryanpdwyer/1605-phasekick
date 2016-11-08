from __future__ import division
from os import path
from tqdm import tqdm
import numpy as np
import phasekick
import phasekickstan as p
import time
from collections import OrderedDict
import lockin
import h5py
import datetime

# Estimated run time 15 min, rMBP 2012

print("tr-EFM Signal averaged curve fitting")
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

outdir = '../results/tr-efm-pystan'

grs = OrderedDict((
    ('151217-200319-p1sun-df',
    {
 'mu_df0': 0,
 'mu_df_inf': -13,
 'mu_tau': np.array([3.,15.]),
 'sigma_df0': 3,
 'sigma_df_inf': 5,
 'sigma_tau': np.array([5., 10.])}
    ),
    ('151217-205007-p3sun-df',
    {
 'mu_df0': 0,
 'mu_df_inf': -15,
 'mu_tau': np.array([1., 5.]),
 'sigma_df0': 3,
 'sigma_df_inf': 5,
 'sigma_tau': np.array([4, 15.])}
    ),
    ('151217-211131-1sun-df',
{
 'mu_df0': 0,
 'mu_df_inf': -15,
 'mu_tau': np.array([0.5,2]),
 'sigma_df0': 3,
 'sigma_df_inf': 5,
 'sigma_tau': np.array([1, 2])}
    ),
    ('151217-234238-20sun-df-384',
{
 'mu_df0': 0,
 'mu_df_inf': -20,
 'mu_tau': np.array([0.1,1]),
 'sigma_df0': 2.5,
 'sigma_df_inf': 7,
 'sigma_tau': np.array([0.5, 2])}
    ),
    ('151218-003450-100sun-784',
{
 'mu_df0': 0,
 'mu_df_inf': -35,
 'mu_tau': np.array([0.1,1]),
 'sigma_df0': 2.5,
 'sigma_df_inf': 8,
 'sigma_tau': np.array([0.5, 2])}
    )
)
)

file = '../results/tr-efm/tr-efm.h5'

start = time.time()
i = 0
total = len(grs)
with h5py.File(file, 'r') as fh:
    for gr_name, priors in grs.items():
        print(datetime.datetime.isoformat(datetime.datetime.now())[11:-7])
        gr = fh[gr_name]
        out_path = path.join(outdir, gr_name+'-dflive_doub.h5')
        data = arrays_to_stan_data(gr['t'].value, gr['mu_df'].value,
                                   gr['sigma_df'].value, gr['N'].value)
        pm = p.PhasekickModel('dflive_doub', data, priors=priors)
        pm.run(chains=4, iter=3000)
        pm.save(out_path)
        diff = time.time() - start
        i += 1
        print('{}/{} Elapsed: {:.1f} s, Remaining: {:.1f} s'.format(i, total, diff, diff/i*(total-i)
            ))

