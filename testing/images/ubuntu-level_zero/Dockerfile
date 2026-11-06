# Ubuntu + Intel GPU (Level Zero) test image. Build context: testing/images/
# Level Zero over Intel's compute-runtime, JIT SPIR-V (spir64) so one build runs
# on integrated Xe and Arc alike. Xe-LP/Xe-LPG integrated GPUs and Arc A-series
# have no FP64 hardware; the two compute-runtime settings below turn on its FP64
# emulation (Xe2: Arc B-series, Lunar Lake, has native FP64). The persistent
# SYCL cache keeps JIT results across processes, so the warm-up carries over.
FROM ubuntu:24.04

ARG DPCPP_VERSION=v7.2.0
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=ubuntu \
    TEST_BACKEND=level_zero \
    BUILD_KIND=sycl \
    SYCL_TARGETS=spir64 \
    SYCL_CACHE_PERSISTENT=1 \
    OverrideDefaultFP64Settings=1 \
    IGC_EnableDPEmulation=1

# Intel's PPA: the compute-runtime in the Ubuntu archive predates Arc B-series.
RUN apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common gpg-agent \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates bzip2 tar \
        libze1 libze-intel-gpu1 intel-opencl-icd \
    && rm -rf /var/lib/apt/lists/*

# Heavy, rarely changing layer: micromamba, DPC++ and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
