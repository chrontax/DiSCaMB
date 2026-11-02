#!/usr/bin/env bash
# OS-independent image provisioning, run once at image build time:
#   1. installs micromamba,
#   2. installs the intel/llvm DPC++ (SYCL) toolchain release,
#   3. creates the two pyDiSCaMB environments: discamb-cpu and discamb-sycl.
# System packages (compiler, git, curl, bzip2, tar) must already be installed.
set -euo pipefail

: "${MAMBA_ROOT_PREFIX:?}" "${DPCPP_ROOT:?}" "${DPCPP_VERSION:?}"
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "==> Installing micromamba"
curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local bin/micromamba
micromamba --version

echo "==> Installing DPC++ ${DPCPP_VERSION} into ${DPCPP_ROOT}"
mkdir -p "$DPCPP_ROOT"
curl -fsSL "https://github.com/intel/llvm/releases/download/${DPCPP_VERSION}/sycl_linux.tar.gz" \
    | tar -xz -C "$DPCPP_ROOT"
"$DPCPP_ROOT/bin/clang++" --version
test -f "$DPCPP_ROOT/lib/libur_adapter_cuda.so" \
    || { echo "DPC++ release lacks the CUDA adapter" >&2; exit 1; }

for kind in cpu sycl; do
    echo "==> Creating micromamba env discamb-${kind}"
    micromamba create -y -n "discamb-${kind}" -f "$here/environment.yml"
done
micromamba clean -a -y
