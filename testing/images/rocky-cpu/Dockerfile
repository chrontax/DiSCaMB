# Rocky Linux test image for the cpu backend: pyDiSCaMB built with the system
# compiler, no SYCL toolchain. Build context: testing/images/
FROM rockylinux/rockylinux:9

ENV MAMBA_ROOT_PREFIX=/opt/micromamba \
    TEST_SYSTEM=rocky \
    TEST_BACKEND=cpu \
    BUILD_KIND=native

RUN dnf install -y --allowerasing \
        gcc gcc-c++ make git curl ca-certificates bzip2 tar findutils \
    && dnf clean all

# Heavy, rarely changing layer: micromamba and the environment.
COPY common/provision.sh common/environment.yml /opt/discamb-testing/
RUN /opt/discamb-testing/provision.sh

COPY common/env-exec common/run-tests.sh /opt/discamb-testing/
WORKDIR /work
ENTRYPOINT ["/opt/discamb-testing/run-tests.sh"]
