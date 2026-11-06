# Rocky Linux + Intel GPU (Level Zero) test image. Build context: testing/images/
# Level Zero over Intel's compute-runtime, JIT SPIR-V (spir64) so one build runs
# on integrated Xe and Arc alike. Xe-LP/Xe-LPG integrated GPUs and Arc A-series
# have no FP64 hardware; the two compute-runtime settings below turn on its FP64
# emulation (Xe2: Arc B-series, Lunar Lake, has native FP64). The persistent
# SYCL cache keeps JIT results across processes, so the warm-up carries over.
FROM rockylinux/rockylinux:9

ARG DPCPP_VERSION=v7.2.0
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=rocky \
    TEST_BACKEND=level_zero \
    BUILD_KIND=sycl \
    SYCL_TARGETS=spir64 \
    SYCL_CACHE_PERSISTENT=1 \
    OverrideDefaultFP64Settings=1 \
    IGC_EnableDPEmulation=1

RUN curl -fsSL -o /etc/yum.repos.d/intel-gpu.repo \
        https://repositories.intel.com/gpu/rhel/9.6/unified/intel-gpu-9.6.repo \
    && dnf install -y --allowerasing \
        gcc gcc-c++ make git curl ca-certificates bzip2 tar findutils \
        level-zero intel-level-zero-gpu intel-opencl \
    && dnf clean all

# Heavy, rarely changing layer: micromamba, DPC++ and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
