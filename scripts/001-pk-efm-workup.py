from __future__ import division
import glob
import time
import h5py
import numpy as np
from scipy import stats
import phasekick

files = glob.glob("../data/pk-efm/*.h5")
outdir = '../results/pk-efm'



now = time.time()
i = 0
itot = len(files)


def expon_weights(tau, fs, coeff_ratio=5.):
    scale = tau * fs
    i = np.arange(int(round(scale * coeff_ratio)))
    return stats.expon.pdf(i, scale=scale)


fs = 1e6
wb = 0.67e-3
wa = 1.2e-3

scale_b = fs*wb
ib = np.arange(int(round(scale_b*5)))  # Use 5 standard deviations worth of data
weight_b = stats.expon.pdf(ib, scale=scale_b)

scale_a = fs*wa
ia = np.arange(int(round(scale_a*5)))
weight_a = stats.expon.pdf(ia, scale=scale_a)

start = time.time()
i = 0
itot = len(files)

for fname in files:
    fh = h5py.File(fname, 'r')
    df, extras = phasekick.workup_adiabatic_w_control_correct_phase_bnc3(
        fh, 2000, 8000, 1e6, w_before=weight_b, w_after=weight_a)
    extras['w_before_ms'] = wb*1e3
    extras['w_after_ms'] = wa*1e3
    phasekick.report_adiabatic_control_phase_corr3(df, extras, outdir=outdir)
    i += 1
    elapsed = time.time() - now
    print("{}/{} complete in {:.2f} min. Estimated {:.1f} min remaining.".format(
        i, itot, elapsed / 60., elapsed / 60. * (itot - i) / i))



