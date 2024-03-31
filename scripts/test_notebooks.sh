#!/bin/bash

pip install ipython
pip install nbformat
pip install seaborn
for nb in examples/*.ipynb; do
    echo "Running notebook smoke test on $nb"
    ipython -c "%run $nb"
done
