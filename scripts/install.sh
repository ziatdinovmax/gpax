#!/bin/bash

pip install toml

if [ "$1" = "test" ]; then
	python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["optional-dependencies"]["test"]))' | pip install -r /dev/stdin

else
	python3 -c 'import toml; c = toml.load("pyproject.toml"); print("\n".join(c["project"]["dependencies"]))' | pip install -r /dev/stdin
fi
