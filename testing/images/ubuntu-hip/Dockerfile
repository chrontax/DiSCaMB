# Ubuntu + ROCm (HIP) test image. Build context: testing/images/
# SYCL_TARGETS (amd_gpu_<gfx>) is passed by test-linux.sh at run time: HIP code
# is compiled ahead of time for the host's GPU architectures. ROCm 6.x because
# the DPC++ HIP adapter links libamdhip64.so.6.
ARG ROCM_VERSION=6.4.4
FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}

# The v7.x Linux releases and nightlies ship no HIP adapter (libur_adapter_hip);
# v6.3.0 is the newest binary release that does.
ARG DPCPP_VERSION=v6.3.0
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=ubuntu \
    TEST_BACKEND=hip \
    BUILD_KIND=sycl

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates bzip2 tar \
    && rm -rf /var/lib/apt/lists/*

# Heavy, rarely changing layer: micromamba, DPC++ and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
