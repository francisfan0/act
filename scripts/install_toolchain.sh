#!/bin/sh
# Install ACT, the standard library, annotate, and actsim into $ACT_HOME.
# Sources are cloned into $ACT_HOME/src. Re-running rebuilds from those clones.
set -eu

ACT_HOME="${ACT_HOME:-$HOME/async}"
export ACT_HOME
mkdir -p "$ACT_HOME/src"
cd "$ACT_HOME/src"

for repo in act stdlib annotate actsim; do
  if [ ! -d "$repo/.git" ]; then
    git clone "https://github.com/asyncvlsi/$repo" "$repo"
  fi
done

export VLSI_TOOLS_SRC="$ACT_HOME/src/act"
cd "$VLSI_TOOLS_SRC"
./configure "$ACT_HOME"
./build
make install

make -C "$ACT_HOME/src/stdlib" install

cd "$ACT_HOME/src/annotate"
./build.sh

cd "$ACT_HOME/src/actsim"
git submodule update --init
./configure
./build.sh

echo "ACT_HOME=$ACT_HOME"
echo "act     $(git -C "$ACT_HOME/src/act" rev-parse HEAD)"
echo "stdlib  $(git -C "$ACT_HOME/src/stdlib" rev-parse HEAD)"
echo "annotate $(git -C "$ACT_HOME/src/annotate" rev-parse HEAD)"
echo "actsim  $(git -C "$ACT_HOME/src/actsim" rev-parse HEAD)"
