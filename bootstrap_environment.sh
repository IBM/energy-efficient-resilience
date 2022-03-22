#!/usr/bin/env sh
#
# energy-efficient-resilience AI support scripts
#

set -e
scriptpath=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )

ver="$1"
if [ -z "$ver" ]; then
    ver=3
fi

python="$(command -v "python$ver" || echo "python$ver not found in path")"
if [ ! -x "$python" ]; then
    echo "$python"
    exit 1
fi

if [ -L "$python" ]; then
    name=$(basename "$(readlink "$python")")
else
    name=$(basename "$python")
fi

rm -fr "$scriptpath/venv"
rm -fr "$scriptpath/venv-$name"

set +e
venvcmd=$(which virtualenv-3)
if [ "$venvcmd" = "" ]; then
    venvcmd=$(which virtualenv)
fi
if [ "$venvcmd" = "" ]; then
    echo "virtualenv support not found"
    exit 1
fi
set -e

if [ -e "$HOME/scratch" ]; then
    rm -f "$HOME/scratch/venv-$name"
    "$venvcmd" "$HOME/scratch/venv-$name" --prompt="(EERAI $name) " --python="$(command -v "python$ver")"
    ln -s "$HOME/scratch/venv-$name" "$scriptpath/venv-$name"
else
    "$venvcmd" "$scriptpath/venv-$name" --prompt="(EERAI $name) " --python="$(command -v "python$ver")"
fi

ln -s "$scriptpath/venv-$name" "$scriptpath/venv"
# shellcheck disable=SC1090
. "$scriptpath/venv-$name/bin/activate"
pip3 install -U pip
pip3 install -U -r requirements_devel.txt
pip3 install -U -r requirements.txt
# shellcheck disable=SC2046
pip3 install -U $(pip3 list | grep "\." | cut -d " " -f 1)
# shellcheck disable=SC2046
pip3 install -U $(pip3 list | grep "\." | cut -d " " -f 1)
# shellcheck disable=SC2046
pip3 install -U $(pip3 list | grep "\." | cut -d " " -f 1)

{
    echo "echo EERAI environment activated"
    echo "$scriptpath/eerai_torch_support.py"
} >> "$scriptpath/venv-$name/bin/activate"
