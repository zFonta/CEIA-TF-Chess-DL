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

# Search one position and report whether real search output came back.
#
# The binary has to *search*, not just start. Piping "quit" on its own is not a
# test: Stockfish exits before evaluating anything, so a build that dies the
# moment it searches still looks fine. stdin is also held open past the `go`,
# because closing it counts as a quit and the engine would exit before
# searching -- the very failure this is meant to catch.
searches_ok() {
    local binary="$1" position="$2" out
    [ -x "${binary}" ] || return 1

    out="$({ printf 'uci\nucinewgame\nisready\nposition %s\ngo depth 8\n' "${position}"
             sleep 5; } | timeout 60 "${binary}" 2>/dev/null)" || return 1

    case "${out}" in *"bestmove"*) ;; *) return 1 ;; esac
    case "${out}" in *"info depth"*) ;; *) return 1 ;; esac
    return 0
}

# Acceptance test: can this build search an ordinary position at all?
#
# Only this decides whether a binary is usable. The bar is deliberately low,
# because the fallbacks swap in a *different* Stockfish, and the exact engine
# is what makes the labels reproducible (requirement 2.3): rejecting a working
# build would trade a known version for an unknown one.
runs_ok() {
    searches_ok "$1" "startpos"
}

# Warning-only probe: some environments crash on sparse endgames.
#
# Not a rejection, because the build labels ordinary positions correctly and
# the pipeline retries then drops what fails. It is worth saying out loud,
# though: silently dropped endgames would skew the dataset.
warn_if_fragile() {
    if ! searches_ok "$1" "fen 8/8/8/4k3/8/8/4Q3/4K3 w - - 0 1"; then
        echo
        echo "WARNING: this build searches normal positions but crashes on a bare"
        echo "         king-and-queen endgame. Labelling will work, but positions"
        echo "         with very little material may be dropped from the dataset."
    fi
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
warn_if_fragile "${TARGET}"
echo
echo "Add it to PATH for this session with:"
echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
