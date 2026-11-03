# Rocky Linux + CUDA test image. Build context: testing/images/
ARG CUDA_VERSION=12.9.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-rockylinux9

ARG DPCPP_VERSION=v7.2.0
ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=rocky \
    TEST_BACKEND=cuda \
    BUILD_KIND=sycl \
    SYCL_TARGETS=nvptx64-nvidia-cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN dnf install -y --allowerasing \
        gcc gcc-c++ make git curl ca-certificates bzip2 tar findutils \
    && dnf clean all

# Heavy, rarely changing layer: micromamba, DPC++ and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
