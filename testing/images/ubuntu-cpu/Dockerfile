# Ubuntu test image for the cpu backend: pyDiSCaMB built with the system
# compiler, no SYCL toolchain. Build context: testing/images/
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    TEST_SYSTEM=ubuntu \
    TEST_BACKEND=cpu \
    BUILD_KIND=native

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates bzip2 tar \
    && rm -rf /var/lib/apt/lists/*

# Heavy, rarely changing layer: micromamba and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
