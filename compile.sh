#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

# Derive the venv's python version and extension suffix rather than hardcoding
# them, so make install and the .so sync follow the venv when it is upgraded.
PY_TAG="$("$VENV_PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
EXT_SUFFIX="$("$VENV_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/$PY_TAG/site-packages"

# Load HPC modules (gcc + MPI + cmake + eigen + fftw)
module load cmake gcc/13.2.0 impi/19.0.9 eigen/3.4.0 fftw3/3.3.10

# Install the python dependencies (just in case they're not already installed)
source .venv/bin/activate
pip install -r "$SCRIPT_DIR/requirements.txt"

# Clean old build (to avoid stale CMake cache)
BUILD_DIR="$SCRIPT_DIR/src/cpp/build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_FFTW=ON \
    -DPYTHON_EXECUTABLE="$VENV_PYTHON" \
    -DCMAKE_INSTALL_PREFIX="$SITE_PACKAGES" \
    -Dpybind11_DIR=$("$VENV_PYTHON" -c "import pybind11; print(pybind11.get_cmake_dir())")

make -j$(nproc)
make install

# Sync freshly built .so into scripts/
for so in "$SITE_PACKAGES/"_*_cpp"$EXT_SUFFIX"; do
    dest="$SCRIPT_DIR/scripts/$(basename "$so")"
    cp -p "$so" "$dest.tmp.$$"
    mv -f "$dest.tmp.$$" "$dest"
done
