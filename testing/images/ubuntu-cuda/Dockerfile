# Ubuntu + CUDA test image. Build context: testing/images/
ARG CUDA_VERSION=12.9.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ARG DPCPP_VERSION=v6.3.0
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    DPCPP_ROOT=/opt/dpcpp \
    DPCPP_VERSION=${DPCPP_VERSION} \
    TEST_SYSTEM=ubuntu \
    TEST_BACKEND=cuda \
    SYCL_TARGETS=nvptx64-nvidia-cuda \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates bzip2 tar \
    && rm -rf /var/lib/apt/lists/*

# Heavy, rarely changing layer: micromamba, DPC++ and both environments.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh common/probe_engine.py /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
