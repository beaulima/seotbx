#!/usr/bin/env bash
CONDA_ENV=seotbx
source /misc/voute1_ptl-bema1/visi/soft/miniconda3/bin/activate ${CONDA_ENV}
conda env update --file ../environment.yml -n seotbx
#conda activate seotbx
#pip install -e ../ --no-deps