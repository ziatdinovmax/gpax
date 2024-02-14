#!/bin/bash

# Good stuff. The poor man's toml parser
# https://github.com/pypa/pip/issues/8049
# This is the analog of pip install -e ".[...]" since for whatever reason
# it does not appear to work cleanly with pip
install_doc_requirements_only () {
    python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["doc"]))' | pip install -r /dev/stdin
}

install_test_requirements_only () {
    python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["test"]))' | pip install -r /dev/stdin
}

install_requirements() {
    python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["dependencies"]))' | pip install -r /dev/stdin
}


pip install toml
if [ "$1" = "doc" ]; then
    install_doc_requirements_only
elif [ "$1" = "test" ]; then
    install_test_requirements_only
else
    install_requirements
fi
