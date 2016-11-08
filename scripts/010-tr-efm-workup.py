"""
The tr-EFM workup used here is not different than the one used previously,
but it is collected here into a single script, rather than being spread 
across ~10 Jupyter notebooks.

Estimated run time: 1 min on rMBP 2012
"""
from __future__ import division
from os import path
from tqdm import tqdm
import numpy as np
import phasekick
from collections import OrderedDict
import lockin
import h5py

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


filesdict = OrderedDict((
    ('../data/tr-efm/151217-200319-p1sun-df.h5',
        (0.1,
        {'fp': 1000,
         'fc': 4000,
         't_phase': -0.052,
         'ti': -0.005,
         'tf': 0.048,
          'Ndec': 62}
          )
    ),
    ('../data/tr-efm/151217-205007-p3sun-df.h5',
        (0.3,
        {'fp': 1000,
         'fc': 4000,
         't_phase': -0.052,
         'ti': -0.002,
         'tf': 0.07,
         'Ndec': 62}
         )
    ),
    ('../data/tr-efm/151217-211131-1sun-df.h5',
        (1.,
        {'fp': 2000,
         'fc': 8000,
         't_phase': -0.052,
         'ti': -2e-3,
         'tf': 25e-3,
         'Ndec': 31}
        )
    ),
    ('../data/tr-efm/151217-234238-20sun-df-384.h5',
        (20.,
        {'fp': 4000,
         'fc': 15000,
         't_phase': -0.052,
         'ti': -500e-6,
         'tf': 2.8e-3,
         'Ndec': 15}
         )
    ),
    ('../data/tr-efm/151218-003450-100sun-784.h5',
        (100.,
        {'fp': 4000,
         'fc': 15000,
         't_phase': -0.052,
         'ti': -500e-6,
         'tf': 1.4e-3,
         'Ndec': 15}
         )
    )
)
)

def main(filesdict, outfile):
    """Save decimated frequency shift data here for curve fitting in PyStan."""

    print("tr-EFM Signal averaged workup")
    print("=============================")
    print("\n")

    files = list(filesdict.keys())
    dsets = sum(len(h5py.File(filename, 'r')['data'].keys()) for filename in files)

    subdict = {key: val for key, val in filesdict.items() if key in files}

    fh_out = h5py.File(outfile, 'w')

    print("Processing data from files:\n{}".format("\n".join(files)))
    with tqdm(total=dsets, leave=True, unit='B', unit_scale=True) as pbar:
        for file, (intensity, params) in tqdm(subdict.items()):
            with h5py.File(file) as fh:
                id = get_filename_without_ext(file)

                trefm = phasekick.AverageTrEFM.from_group(fh['data'],
                                                  params['fp'],
                                                  params['fc'],
                                                  params['t_phase'],
                                                  params['ti'],
                                                  params['tf'],
                                                  pbar=pbar)

                t, mu_df, sigma_df = get_t_mu_sigma_df(trefm)
                N = trefm.t.shape[0]
                gr = fh_out.create_group(id)

                Ndec = params['Ndec']

                save_workup(gr, params, t[::Ndec],
                            mu_df[::Ndec], sigma_df[::Ndec], intensity, N)


    fh_out.close()

if __name__ == '__main__':
    main(filesdict, '../results/tr-efm/tr-efm.h5')
