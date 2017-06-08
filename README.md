Microsecond photocapacitance transients data and workup
=======================================================

This repository contains the data and workup for the paper, "Microsecond photocapacitance transients observed using a charged microcantilever as a gated
mechanical integrator," by Ryan P. Dwyer, Sarah R. Nathan, and John A. Marohn.

Reproducing this analysis
-------------------------

### Installing dependencies

To reproduce this analysis in *any* Python environment, run the command

    pip install -r requirements.txt

This will install the custom packages needed for the data workup.
It is best if large packages containing compiled code like `numpy`,
`scipy`, `h5py`, and `pystan` are installed beforehand using a
package manager such as `conda` or `Canopy`.


To reproduce this analysis in a dedicated [conda](https://www.continuum.io/downloads) environment, install `conda` and run the command

    bash conda-env.sh ENVIRONMENT_NAME

where `ENVIRONMENT_NAME` is the name of the `conda` enviornment that will be created.


### Running the analysis

The analysis requires a significant amount of time (30 to 90 minutes) and storage (3.5 GB).
To download the experimental data, analysis the data, and generate paper figures, run 

    bash all-analysis.sh | tee logfile.txt

or if you need to activate the conda environment

    source activate ENVIRONMENT_NAME
    bash all-analysis.sh | tee logfile.txt

The second half of the command mirrors the output to the file `logfile.txt`. The data analysis Python scripts are in the folder `scripts`, the scripts or IPython notebooks for generating the figures are in the folder `figs_scripts`, and the generated figures are placed in the folder `figs`.


PyStan Notes
------------

I ran into issues with `pystan` finding gcc first on the path,
but still linking against Mac's clang `libstdc++`.
To fix, I linked Mac's default compilers into `/usr/local/bin`,
and added that first to my path:

    ln -s /usr/bin/gcc /usr/local/bin/gcc
    ln -s /usr/bin/g++ /usr/local/bin/g++
    export PATH="/usr/local/bin:$PATH"



