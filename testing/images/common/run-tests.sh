#!/usr/bin/env bash
# Container entrypoint: prepares the pyDiSCaMB source, builds it into the
# discamb environment (BUILD_KIND=native: system compiler; sycl: DPC++ for
# $SYCL_TARGETS), checks that it imports and, for sycl, that the SYCL runtime
# sees a $TEST_BACKEND device, then runs pytest over lib/discamb/testing/tests.
#
# Inputs (environment):
#   SOURCE_MODE    local (default): copy the pyDiSCaMB tree mounted at /src
#                  git: clone PYDISCAMB_REPO@PYDISCAMB_REF and replace the
#                       lib/discamb submodule with DISCAMB_REPO@DISCAMB_REF
#   PYTEST_ARGS    extra arguments passed to pytest (word-split)
#   DEBUG          1: also print the setup log (always written to
#                  /results/setup.log); otherwise only pytest output and errors
#   HOST_UID/GID   owner given to files written to /results
set -euo pipefail

SOURCE_MODE=${SOURCE_MODE:-local}
PYDISCAMB_REPO=${PYDISCAMB_REPO:-https://github.com/discamb-project/pyDiSCaMB.git}
PYDISCAMB_REF=${PYDISCAMB_REF:-main}
DISCAMB_REPO=${DISCAMB_REPO:-https://github.com/chrontax/DiSCaMB.git}
DISCAMB_REF=${DISCAMB_REF:-main}
PYTEST_ARGS=${PYTEST_ARGS:-}
DEBUG=${DEBUG:-0}

ENV_EXEC=/opt/discamb-testing/env-exec
ENV_PREFIX="$MAMBA_ROOT_PREFIX/envs/discamb"
SRC=/work/src
BUILD=/work/build
RESULTS=/results
SETUP_LOG="$RESULTS/setup.log"
TAG="$TEST_SYSTEM-$TEST_BACKEND"

log() {
    local line
    line=$(printf '[%s %s] %s' "$(date +%H:%M:%S)" "$TAG" "$*")
    printf '%s\n' "$line" >>"$SETUP_LOG"
    if [ "$DEBUG" = 1 ]; then printf '%s\n' "$line"; fi
}
die() {
    printf '[%s] ERROR: %s\n' "$TAG" "$*" >>"$SETUP_LOG"
    printf '[%s] ERROR: %s (setup log: setup.log)\n' "$TAG" "$*" >&2
    exit 1
}

# Only when the mount shows the host user as owner (rootful docker): under
# rootless docker/podman container root already is the host user. Never fatal,
# so a failed chown cannot replace the test result as the exit status.
fix_results_owner() {
    if [ -n "${HOST_UID:-}" ] && [ -d "$RESULTS" ] \
        && [ "$(stat -c %u "$RESULTS")" = "$HOST_UID" ]; then
        chown -R "$HOST_UID:${HOST_GID:-$HOST_UID}" "$RESULTS" \
            || printf '[%s] WARNING: could not chown %s to %s\n' "$TAG" "$RESULTS" "$HOST_UID" >&2
    fi
}
trap fix_results_owner EXIT

prepare_source() {
    case "$SOURCE_MODE" in
        local)
            [ -f /src/pyproject.toml ] || die "no pyDiSCaMB tree mounted at /src"
            log "copying local source tree from /src"
            mkdir -p "$SRC"
            tar -C /src \
                --exclude=./build \
                --exclude=./lib/discamb/build \
                --exclude=./lib/discamb/testing/results \
                --exclude=.cache \
                --exclude=__pycache__ \
                -cf - . | tar -C "$SRC" -xf -
            ;;
        git)
            log "cloning $PYDISCAMB_REPO@$PYDISCAMB_REF"
            git clone --quiet "$PYDISCAMB_REPO" "$SRC"
            git -C "$SRC" checkout --quiet "$PYDISCAMB_REF"
            git -C "$SRC" submodule set-url lib/discamb "$DISCAMB_REPO"
            git -C "$SRC" submodule update --quiet --init --recursive data/MATTS lib/pybind11_json
            log "replacing lib/discamb with $DISCAMB_REPO@$DISCAMB_REF"
            rm -rf "$SRC/lib/discamb"
            git clone --quiet "$DISCAMB_REPO" "$SRC/lib/discamb"
            git -C "$SRC/lib/discamb" checkout --quiet "$DISCAMB_REF"
            ;;
        *)
            die "unknown SOURCE_MODE '$SOURCE_MODE' (expected local or git)"
            ;;
    esac
    grep -q 'SYCL' "$SRC/lib/discamb/src/Scattering/CMakeLists.txt" \
        || die "lib/discamb has no SYCL build option; is it the chrontax/DiSCaMB fork?"
}

build_discamb() {
    local logfile="$RESULTS/build.log"
    local -a args=(-v "$SRC" -C "build-dir=$BUILD"
        -C "cmake.define.DISCAMB_BUILD_TESTS=OFF")
    local -a extra_env=()

    if [ "$BUILD_KIND" = sycl ]; then
        args+=(
            -C "cmake.define.CMAKE_CXX_COMPILER=$DPCPP_ROOT/bin/clang++"
            -C "cmake.define.SYCL=ON"
            -C "cmake.define.TAAM_WITH_GPU_IMP_26=ON"
            -C "cmake.define.SYCL_INCLUDE_DIR=$DPCPP_ROOT/include"
            -C "cmake.define.SYCL_LIBRARY_DIR=$DPCPP_ROOT/lib"
            -C "cmake.define.SYCL_TARGETS=$SYCL_TARGETS"
            -C "cmake.define.CMAKE_CXX_FLAGS=-I$ENV_PREFIX/include"
        )
        # clang links libomp from the env; it is not on the system linker path.
        extra_env=(LIBRARY_PATH="$ENV_PREFIX/lib")
    fi

    log "building pyDiSCaMB ($BUILD_KIND) (log: $(basename "$logfile"))"
    local start=$SECONDS
    if ! env "${extra_env[@]}" "$ENV_EXEC" python -m pip install "${args[@]}" >"$logfile" 2>&1; then
        tail -n 60 "$logfile" >&2
        die "pyDiSCaMB ($BUILD_KIND) build failed"
    fi
    log "pyDiSCaMB ($BUILD_KIND) built in $((SECONDS - start))s"
}

check_build() {
    "$ENV_EXEC" python -c 'import pydiscamb' || die "pydiscamb does not import"
    log "pydiscamb imports"
    [ "$BUILD_KIND" = sycl ] || return 0

    local devices
    devices=$("$ENV_EXEC" "$DPCPP_ROOT/bin/sycl-ls" 2>&1) \
        || die "sycl-ls failed: $devices"
    log "sycl-ls:"$'\n'"$devices"
    grep -q "^\[$TEST_BACKEND:" <<<"$devices" \
        || die "SYCL runtime sees no $TEST_BACKEND device (is the GPU passed to the container?)"
}

run_tests() {
    local tests_dir="$SRC/lib/discamb/testing/tests"
    [ -d "$tests_dir" ] || die "test directory $tests_dir does not exist"

    export DISCAMB_TEST_ENV_EXEC="$ENV_EXEC"
    export DISCAMB_TEST_SYSTEM="$TEST_SYSTEM"
    export DISCAMB_TEST_BACKEND="$TEST_BACKEND"
    export DISCAMB_TEST_BUILD="$BUILD_KIND"
    export DISCAMB_TEST_RESULTS="$RESULTS"
    export DISCAMB_TEST_SOURCE="$SRC"

    local -a extra=()
    read -r -a extra <<<"$PYTEST_ARGS"

    log "running pytest in $tests_dir"
    local rc=0
    (cd "$tests_dir" && "$ENV_EXEC" python -m pytest -v -s -p no:cacheprovider -rA \
        --junitxml="$RESULTS/junit.xml" "${extra[@]}" .) || rc=$?
    # pytest exit code 5 means no tests were collected.
    if [ "$rc" -eq 5 ]; then
        printf '[%s] no tests collected\n' "$TAG"
        rc=0
    fi
    return "$rc"
}

mkdir -p "$RESULTS"
log "build: $BUILD_KIND${DPCPP_VERSION:+ (DPC++ $DPCPP_VERSION)}, source mode: $SOURCE_MODE"
prepare_source
build_discamb
check_build
run_tests
log "all tests passed"
