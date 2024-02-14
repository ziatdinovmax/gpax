#!/bin/bash

build_docs () {

    if [[ "${GITHUB_ACTION_IS_RUNNING}" = 1 ]]; then
        bash scripts/install.sh doc
    fi

    make -C docs/ html

    # Helper when running on local. If not running in a GitHub Actions
    # environment, this will attempt to open index.html with the users'
    # default program
    if [[ -z "${GITHUB_ACTION_IS_RUNNING}" ]]; then
        open docs/build/html/index.html
    fi
    
}

pip install toml
bash scripts/install.sh
bash scripts/install.sh doc
bash scripts/update_version.sh set
build_docs
bash scripts/update_version.sh reset
