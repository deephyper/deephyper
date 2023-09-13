#!/bin/bash
#COBALT -n 1
#COBALT -q full-node
#COBALT -t 120

. /etc/profile

# Generic installation script for DeepHyper on ALCF's Polaris.
# This script is meant to be run on the login node of the machine.
# It will install DeepHyper and its dependencies in the current directory.
# A good practice is to create a `build` folder and launch the script from there,
# e.g. from the root of the DeepHyper repository:
# $ mkdir build && cd build && cp ../install/alcf/thetagpu.sh install.sh
# $ module load cobalt/cobalt-gpu
# $ qsub -A $PROJECT install.sh
# The script will also create a file named `activate-dhenv.sh` that will
# Setup the environment each time it is sourced `source activate-dhenv.sh`.

set -x

# Load modules available on the current system
module load llvm/main-20220317
module load conda/2023-01-11

# Copy the base conda environment
conda create -p dhenv --clone base -y
conda activate dhenv/
pip install --upgrade pip

# Install RedisJSON with Spack
# Install Spack
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh

git clone https://github.com/deephyper/deephyper-spack-packages.git

# Create and activate the `redisjson` environment
spack env create redisjson
spack env activate redisjson

# Add the DeepHyper Spack packages to the environment
spack repo add deephyper-spack-packages

# Add the `redisjson` Spack package to the environment
spack add redisjson

# Build the environment
spack install

# Install mpi4py
git clone https://github.com/mpi4py/mpi4py.git
pushd mpi4py/
MPICC=mpicc python setup.py install
popd

# Install the DeepHyper's Python package
git clone -b master https://github.com/deephyper/deephyper.git

# Install DeepHyper with MPI and Redis backends
pip install -e "deephyper/[default,mpi,redis-hiredis]"

# Create activation script
touch activate-dhenv.sh
echo "#!/bin/bash" >> activate-dhenv.sh

# Append modules loading and conda activation
echo "" >> activate-dhenv.sh
echo "module load llvm/main-20220317" >> activate-dhenv.sh
echo "module load conda/2021-11-30" >> activate-dhenv.sh
echo "module load openmpi/openmpi-4.0.5" >> activate-dhenv.sh
echo "conda activate $PWD/dhenv/" >> activate-dhenv.sh

# Append Spack activation
echo "" >> activate-dhenv.sh
echo ". $PWD/spack/share/spack/setup-env.sh" >> activate-dhenv.sh
echo "spack env activate redisjson" >> activate-dhenv.sh

# Create Redis configuration
touch redis.conf
echo "bind 0.0.0.0" >> redis.conf
cat $(spack find --path redisjson | grep -o "/.*/redisjson.*")/redis.conf >> redis.conf