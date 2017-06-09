from __future__ import division
import os
from tqdm import tqdm
import time
import copy
import glob
import numpy as np
import phasekickstan as p

files = glob.glob("../results/pk-efm-pystan/*.h5")

m_exp = 'exp2_sq_nc'

now = time.time()
i = 0
itot = len(files)
for fname in files:
    basename = os.path.splitext(fname)[0]


    folder, filename = os.path.split(basename)

    print(fname)
    print(filename)

    print(time.strftime("%H:%M:%S",time.localtime()))
    print("\n\n")

    pm = p.PlotStanModels(filename, [fname])
    pm.report(outfile=os.path.join(folder, 'reports/', filename))

    i += 1
    elapsed = time.time() - now

    print("{}/{} complete in {:.2f} min. Estimated {:.1f} min remaining.".format(
        i, itot, elapsed / 60., elapsed / 60. * (itot - i) / i))