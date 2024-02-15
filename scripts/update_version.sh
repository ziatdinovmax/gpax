#!/bin/bash

PACKAGE_NAME="gpax"
SEMANTIC_PLACEHOLDER="...  # semantic-version-placeholder"
INIT_FILE_NAME="$PACKAGE_NAME"/__init__.py



replace_version_in_init () {
    pip install dunamai~=1.12
    version="$(dunamai from git --style pep440 --no-metadata)"
    dunamai check "$version" --style pep440
    sed_command="s/'$SEMANTIC_PLACEHOLDER'/'$version'/g"
    echo "$sed_command"
    cp "$INIT_FILE_NAME" "$INIT_FILE_NAME".bak
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$sed_command" "$INIT_FILE_NAME"
    else
        sed -i "$sed_command" "$INIT_FILE_NAME"
    fi
    echo "__init__ version set to" "$version"
    export _TMP_VERSION="$version"
}

reset_version_to_ellipsis () {
    # we only grep the first instance of __version__
    current_version=$(grep -m 1 "__version__" "$INIT_FILE_NAME"/__init__.py)
    cp "$INIT_FILE_NAME".bak "$INIT_FILE_NAME"
    rm "$INIT_FILE_NAME".bak
    echo "$current_version" "reset to placeholder"
}


if [ "$1" == "set" ]; then
    replace_version_in_init
elif [ "$1" == "reset" ]; then
    reset_version_to_ellipsis
fi
