#!/bin/sh
conda env create -f environment.yml
conda install cython
pip install -r requirements.txt
wget https://services.healthtech.dtu.dk/services/NetSolP-1.0/netsolp-1.0.ALL.tar.gz
mkdir netsolp
tar -zxvf netsolp-1.0.ALL.tar.gz -C netsolp
mv netsolp-1.0.ALL.tar.gz netsolp
cd netsolp/
pip install -r requirements.txt
