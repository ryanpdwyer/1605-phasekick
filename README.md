Microsecond photocapacitance transients data and workup
=======================================================

[![DOI](https://zenodo.org/badge/67623824.svg)](https://zenodo.org/badge/latestdoi/67623824)

This repository contains the data and workup for the paper, "Microsecond photocapacitance transients observed using a charged microcantilever as a gated
mechanical integrator," by Ryan P. Dwyer, Sarah R. Nathan, and John A. Marohn [[DOI: 10.1126/sciadv.1602951](http://dx.doi.org/10.1126/sciadv.1602951)].

Reproducing this analysis
-------------------------

### Existing Python environment

To perform the analysis in an existing Python environment, install the dependencies by running the command

    pip install -r requirements.txt

or for fixed versions of the packages
    
    pip install -r requirements-frozen.txt

This will install the packages needed for the data workup.
It is best if large packages containing compiled code are installed beforehand using a package manager such as `conda` or `Canopy`. The binary packages required are

    numpy
    scipy
    matplotlib
    pandas
    h5py
    pystan
    scikit-learn
    docutils
    lxml

The analysis requires a significant amount of time (30 to 90 minutes) and storage (3.5 GB).
To download the experimental data, analysis the data, and generate paper figures, run 

    bash all-analysis.sh | tee log.txt

The second half of the command mirrors the output to the file `log.txt`. 

### New conda environment

To reproduce this analysis in a dedicated [conda](https://www.continuum.io/downloads) environment, install `conda` and run the command

    bash conda-env.sh ENVIRONMENT_NAME

where `ENVIRONMENT_NAME` is the name of the `conda` environment that will be created.

The analysis requires a significant amount of time (30 to 90 minutes) and storage (3.5 GB).
To download the experimental data, analysis the data, and generate paper figures, run 

    source activate ENVIRONMENT_NAME
    bash all-analysis.sh | tee logfile.txt

The second half of the command mirrors the output to the file `log.txt`. 

Other information
-----------------

### Directory structure


`scripts` — The data analysis Python scripts.

`figs_scripts` — The scripts or IPython notebooks for generating the figures.

`figs` — The generated figures.

`results` – Results from processing the raw experimental data. This directory contains additional `html` reports about each of the pk-EFM data sets.


### PyStan Notes

I ran into issues with `pystan` finding gcc first on the path,
but still linking against Mac's clang `libstdc++`.
To fix, I linked Mac's default compilers into `/usr/local/bin`,
and added that first to my path:

    ln -s /usr/bin/gcc /usr/local/bin/gcc
    ln -s /usr/bin/g++ /usr/local/bin/g++
    export PATH="/usr/local/bin:$PATH"



Package versions used for the paper figure analysis
---------------------------------------------------

Information about the specific environment on Ryan Dwyer's Macbook Pro, the computer used to run the analysis, is in the files `env-conda.txt`,  `env-conda.json`, and `env-pip.txt`.
