from __future__ import division
import os
from tqdm import tqdm
import time
import copy
import numpy as np
import phasekickstan as p

files = {
'../results/pk-efm/151217-201951-p1sun-phasekick.csv': 0.1,
'../results/pk-efm/151217-205912-p3sun-phasekick.csv': 0.3,
'../results/pk-efm/151217-214045-1sun-phasekick.csv': 1,
'../results/pk-efm/151217-220252-1sun-phasekick-shorter.csv': 1,
'../results/pk-efm/151217-233507-20sun-phasekick.csv': 20,
'../results/pk-efm/151218-001254-20sun-phasekick-short.csv': 20,
'../results/pk-efm/151218-002059-20sun-phasekick-short.csv': 20,
'../results/pk-efm/151218-004818-100sun-phasekick.csv': 100,
'../results/pk-efm/151218-011055-100sun-phasekick-768.csv': 100,
'../results/pk-efm/151218-012858-20sun-phasekick-768.csv': 20,
}

def tau_prior(intensity):
    return 0.18 + 0.6 / intensity

def df_prior(intensity):
    return -13 + -0.2 * intensity

exp_priors = {
    'mu_tau': 2.0,
    'sigma_tau': 5.0,
    'mu_df_inf': -20,
    'sigma_df_inf': 15,
}


m_exp = 'exp_sq_no_control'

outdir = '../results/pk-efm-pystan/'

def main(files, chains=4, iterations=10):
    now = time.time()
    i = 0
    itot = len(files)

    for fname, intensity in files.items():
        basename = os.path.splitext(fname)[0]
        pk1 = p.PhasekickModel(m_exp, fname)

        folder, filename = os.path.split(basename)

        print(basename)
        print("\n\n")

        print(time.strftime("%H:%M:%S",time.localtime()))
        print("\n\n")

        new_priors = copy.copy(exp_priors)

        new_priors['mu_tau'] = exp_priors['mu_tau'] * tau_prior(intensity)
        new_priors['sigma_tau'] = exp_priors['sigma_tau']
        new_priors['mu_df_inf'] = df_prior(intensity)


        pk1.run(chains=chains, iter=iterations, priors=new_priors)
        sample_fname = os.path.join(outdir, 'test11'+filename+'_'+'exp_sq_nc'+'.h5')
        pk1.save(sample_fname)

        i += 1
        elapsed = time.time() - now

        print("{}/{} complete in {:.2f} min. Estimated {:.1f} min remaining.".format(
            i, itot, elapsed / 60., elapsed / 60. * (itot - i) / i))

if __name__ == '__main__':
    import sys
    argc = len(sys.argv)
    
    iterations = 10
    chains = 1

    if argc > 3:
        raise ValueError("Too many arguments")
    if argc > 2:
        iterations = int(sys.argv[2])
    if argc > 1:
        if ('-h' in str(sys.argv[1])):
            print("Usage: python 102-pk-efm-stan-single.py [chains] [iterations]")
            sys.exit()
        chains = int(sys.argv[1])

    main(files, chains, iterations)
