#!/bin/bash

#!/bin/bash
for nb in examples/*.ipynb; do
    echo "Running notebook smoke test on $nb"
    ipython -c "%run $nb"
done
