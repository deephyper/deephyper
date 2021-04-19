#!/bin/bash -x

module load postgresql
module load miniconda-3
conda create -p dh-theta python=3.8 -y
conda activate dh-theta/
conda install gxx_linux-64 gcc_linux-64 -y
# DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
pip install deephyper[analytics,balsam]
conda install tensorflow -c intel -y


# Checking existence of "bashrc_theta"
BASHRC_THETA=~/.bashrc_theta

read -r -d '' NEW_BASHRC_CONTENT <<- EOM
# Added by DeepHyper
if [[ $(echo '$HOSTNAME') = *"thetalogin"* ]]; then
    source ~/.bashrc_theta
fi
EOM

if test -f "$BASHRC_THETA"; then
    echo "$BASHRC_THETA exists."
else
    echo "$BASHRC_THETA does not exists."
    echo "Adding new lines to ~/.bashrc"
    echo "$NEW_BASHRC_CONTENT" >> ~/.bashrc
fi


read -r -d '' NEW_BASHRC_THETA_CONTENT <<- EOM
# Added by DeepHyper
module load postgresql
module load miniconda-3
EOM

echo "Adding new lines to $BASHRC_THETA"
echo "$NEW_BASHRC_THETA_CONTENT" >> $BASHRC_THETA
