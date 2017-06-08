#!/bin/sh
if [[ $# -ne 1 ]]; then
    echo "$#"
    echo "sh conda-env.sh ENVIRONMENT_NAME"
    echo "Pass an environment name to create a conda environment for performing
this analysis. Install conda first at https://www.continuum.io/downloads.

To use your current Python environment, just run the command

    pip install -r requirements.txt

to install necessary requirements.
"
    exit 1
fi

conda create -n $1 python=2.7 pip numpy scipy matplotlib nose pandas pystan h5py docutils ipython jupyter notebook lxml scikit-learn

source activate $1
pip install -r requirements.txt
