#!/usr/bin/env bash
#
# Install a pinned Stockfish build and print its version.
#
# The version is pinned rather than taken from the distro package, because the
# dataset's labels are only reproducible if the engine that produced them is
# known exactly (requirement 2.3). The version is also recorded on every row of
# the dataset at build time.
#
# Usage:
#   scripts/setup_stockfish.sh [install_dir]
#
# Environment:
#   SF_VERSION  Stockfish release tag (default: sf_17.1)
#   SF_ARCH     Build variant (default: x86-64-avx2, falls back to x86-64)

set -euo pipefail

SF_VERSION="${SF_VERSION:-sf_17.1}"
SF_ARCH="${SF_ARCH:-x86-64-avx2}"
INSTALL_DIR="${1:-$(pwd)/bin}"
BASE_URL="https://github.com/official-stockfish/Stockfish/releases/download"

mkdir -p "${INSTALL_DIR}"
TARGET="${INSTALL_DIR}/stockfish"

runs_ok() {
    [ -x "$1" ] && echo "quit" | "$1" > /dev/null 2>&1
}

install_release() {
    local arch="$1"
    local archive="stockfish-ubuntu-${arch}.tar"
    local url="${BASE_URL}/${SF_VERSION}/${archive}"
    local tmp
    tmp="$(mktemp -d)"

    echo "Downloading ${url}"
    if ! curl -fsSL "${url}" -o "${tmp}/${archive}"; then
        rm -rf "${tmp}"
        return 1
    fi

    tar -xf "${tmp}/${archive}" -C "${tmp}"
    local binary
    binary="$(find "${tmp}" -type f -name 'stockfish*' -perm -u+x | head -n 1)"
    if [ -z "${binary}" ]; then
        rm -rf "${tmp}"
        return 1
    fi

    cp "${binary}" "${TARGET}"
    chmod +x "${TARGET}"
    rm -rf "${tmp}"

    # An AVX2 build dies with SIGILL on a CPU without AVX2, so the binary is
    # actually run before it is accepted.
    if ! runs_ok "${TARGET}"; then
        rm -f "${TARGET}"
        return 1
    fi
    return 0
}

if runs_ok "${TARGET}"; then
    echo "Stockfish already installed at ${TARGET}"
else
    if ! install_release "${SF_ARCH}"; then
        echo "The ${SF_ARCH} build did not work here; trying the portable x86-64 build."
        if ! install_release "x86-64"; then
            echo "Release download failed; falling back to the distro package."
            if command -v apt-get > /dev/null 2>&1; then
                apt-get update -qq && apt-get install -y -qq stockfish
                ln -sf "$(command -v stockfish)" "${TARGET}"
            else
                echo "ERROR: could not install Stockfish." >&2
                exit 1
            fi
        fi
    fi
fi

echo
echo "Installed at: ${TARGET}"
printf 'uci\nquit\n' | "${TARGET}" | grep '^id name' || true
echo
echo "Add it to PATH for this session with:"
echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
