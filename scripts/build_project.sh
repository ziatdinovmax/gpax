#!/bin/bash

pip install flit~=3.7
pip install dunamai==1.19.2
echo "__version__ = '$(dunamai from any --style=pep440 --no-metadata)'" >gpax/_version.py
flit build
