#!/bin/bash

pip install flit~=3.7
bash scripts/update_version.sh set
flit build
bash scripts/update_version.sh reset
