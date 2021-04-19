#!/bin/bash -x

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O miniconda.sh
bash $PWD/miniconda.sh -b -p $PWD/miniconda
rm -f miniconda.sh

# Install Postgresql
wget http://get.enterprisedb.com/postgresql/postgresql-9.6.13-4-linux-x64-binaries.tar.gz -O postgresql.tar.gz
tar -xf postgresql.tar.gz
rm -f postgresql.tar.gz

source $PWD/miniconda/bin/activate
conda create -p dh-cooley python=3.8 -y
conda activate dh-cooley/
conda install gxx_linux-64 gcc_linux-64 -y
# DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
pip install deephyper[analytics,balsam]
conda install tensorflow-gpu -y

# Checking existence of "bashrc_theta"
BASHRC_COOLEY=~/.bashrc_cooley

read -r -d '' NEW_BASHRC_CONTENT <<- EOM
# Added by DeepHyper
if [[ $(echo '$HOSTNAME') = *"cooley"* ]]; then
    source ~/.bashrc_cooley
fi

if [[ $(echo '$HOSTNAME') = *"cc"* ]]; then
    source ~/.bashrc_cooley
fi
EOM

if test -f "$BASHRC_COOLEY"; then
    echo "$BASHRC_COOLEY exists."
else
    echo "$BASHRC_COOLEY does not exists."
    echo "Adding new lines to ~/.bashrc"
    echo "$NEW_BASHRC_CONTENT" >> ~/.bashrc
fi


echo "Adding new lines to $BASHRC_COOLEY"
echo "# Added by DeepHyper" >> $BASHRC_COOLEY
echo "source $PWD/miniconda/bin/activate" >> $BASHRC_COOLEY
echo "export PATH=$PWD/pgsql/bin:"'$PATH' >> $BASHRC_COOLEY
