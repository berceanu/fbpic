#!/bin/bash

# Exit on first error encountered #############################################
#
set -eu -o pipefail


# Check: ######################################################################
#
#   Was karolina_synchrad.profile sourced and configured correctly?
if [ -z ${proj-} ]; then echo "WARNING: The 'proj' variable is not yet set in your karolina_synchrad.profile file! Please edit its line 2 to continue!"; exit 1; fi


# Remove old dependencies #####################################################
#
SW_DIR="${HOME}/sw/karolina/gpu"
rm -rf ${SW_DIR}
mkdir -p ${SW_DIR}

# remove common user mistakes in python, located in .local instead of a venv
python3 -m pip uninstall -qq -y synchrad
python3 -m pip uninstall -qqq -y mpi4py 2>/dev/null || true


# Python ######################################################################
#
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
rm -rf ${SW_DIR}/venvs/synchrad
python3 -m venv ${SW_DIR}/venvs/synchrad
source ${SW_DIR}/venvs/synchrad/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade black 
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade openpmd-viewer
python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade numba
python3 -m pip install --upgrade mako
python3 -m pip install --upgrade pyopencl