#!/bin/bash

PACKAGE_NAME="gpax"

replace_version_in_init () {
    pip install dunamai~=1.12
    version="$(dunamai from git --style pep440 --no-metadata)"
    dunamai check "$version" --style pep440
    sed_command="s/...  # semantic-version-placeholder/'$version'/g"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$sed_command" "$PACKAGE_NAME"/__init__.py
    else
        sed -i "$sed_command" "$PACKAGE_NAME"/__init__.py
    fi
    echo "__init__ version set to" "$version"
    export _TMP_VERSION="$version"
}

reset_version_to_ellipsis () {
    # we only grep the first instance of __version__
    current_version=$(grep -m 1 "__version__" "$PACKAGE_NAME"/__init__.py)
    sed_command="s/$current_version/__version__ = ...  # semantic-version-placeholder/g"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$sed_command" "$PACKAGE_NAME"/__init__.py
    else
        sed -i "$sed_command" "$PACKAGE_NAME"/__init__.py
    fi
    echo "$current_version" "reset to placeholder"
}


if [ "$1" == "set" ]; then
    replace_version_in_init
elif [ "$1" == "reset" ]; then
    reset_version_to_ellipsis
fi
