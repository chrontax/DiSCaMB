# Rocky Linux + ROCm (HIP) test image. Build context: testing/images/
# SYCL_TARGETS (amd_gpu_<gfx>) is passed by test-linux.sh at run time: HIP code
# is compiled ahead of time for the host's GPU architectures. ROCm 6.x because
# the DPC++ HIP adapter links libamdhip64.so.6.
FROM rockylinux/rockylinux:9

ARG ROCM_VERSION=6.4.4
# The v7.x Linux releases and nightlies ship no HIP adapter (libur_adapter_hip);
# v6.3.0 is the newest binary release that does.
ARG DPCPP_VERSION=v6.3.0
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=rocky \
    TEST_BACKEND=hip \
    BUILD_KIND=sycl

RUN printf '[rocm]\nname=ROCm %s\nbaseurl=https://repo.radeon.com/rocm/el9/%s/main\nenabled=1\ngpgcheck=1\ngpgkey=https://repo.radeon.com/rocm/rocm.gpg.key\n' \
        "$ROCM_VERSION" "$ROCM_VERSION" >/etc/yum.repos.d/rocm.repo \
    && dnf install -y --allowerasing \
        gcc gcc-c++ make git curl ca-certificates bzip2 tar findutils \
        hip-runtime-amd rocm-device-libs rocminfo \
    && dnf clean all \
    && echo /opt/rocm/lib >/etc/ld.so.conf.d/rocm.conf && ldconfig

# Heavy, rarely changing layer: micromamba, DPC++ and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
