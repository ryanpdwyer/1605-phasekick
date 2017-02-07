Phasekick paper data and workup
===============================

This repository contains the data and workup for the paper, TITLE.

Reproducing this analysis
-------------------------

To reproduce this analysis in any Python environment, run the command:

    pip install -r requirements.txt

This will install the custom packages needed for the data workup.
It is best if large packages containing compiled code like `numpy`,
`scipy`, `h5py`, and `pystan` are installed beforehand using a
package manager such as `conda` or `Canopy`.



PyStan Notes
------------

I ran into issues with `pystan` finding gcc first on the path,
but still linking against Mac's clang `libstdc++`.
To fix, I linked Mac's default compilers into `/usr/local/bin`,
and added that first to my path:

    ln -s /usr/bin/gcc /usr/local/bin/gcc
    ln -s /usr/bin/g++ /usr/local/bin/g++
    export PATH="/usr/local/bin:$PATH"



