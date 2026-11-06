#!/usr/bin/env bash
# OS-independent image provisioning, run once at image build time:
#   1. installs micromamba,
#   2. for BUILD_KIND=sycl, installs the intel/llvm DPC++ (SYCL) toolchain release,
#   3. creates the pyDiSCaMB environment `discamb`.
# System packages (compiler, git, curl, bzip2, tar) must already be installed.
set -euo pipefail

: "${MAMBA_ROOT_PREFIX:?}" "${BUILD_KIND:?}" "${TEST_BACKEND:?}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "==> Installing micromamba"
curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local bin/micromamba
micromamba --version

case "$BUILD_KIND" in
    sycl)
        : "${DPCPP_ROOT:?}" "${DPCPP_VERSION:?}"
        echo "==> Installing DPC++ ${DPCPP_VERSION} into ${DPCPP_ROOT}"
        mkdir -p "$DPCPP_ROOT"
        curl -fsSL "https://github.com/intel/llvm/releases/download/${DPCPP_VERSION}/sycl_linux.tar.gz" \
            | tar -xz -C "$DPCPP_ROOT"
        "$DPCPP_ROOT/bin/clang++" --version
        test -f "$DPCPP_ROOT/lib/libur_adapter_${TEST_BACKEND}.so" \
            || { echo "DPC++ release lacks the ${TEST_BACKEND} adapter" >&2; exit 1; }
        ;;
    native) ;;
    *) echo "unknown BUILD_KIND '$BUILD_KIND' (expected native or sycl)" >&2; exit 1 ;;
esac

echo "==> Creating micromamba env discamb"
micromamba create -y -n discamb -f "$here/environment.yml"
micromamba clean -a -y
