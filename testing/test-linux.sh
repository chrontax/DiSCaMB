#!/usr/bin/env bash
# Runs the discamb test suite in docker containers: one container per
# (linux system x GPU backend) pair, for every backend whose GPU is present on
# this machine. Installs nothing on the host; requires docker and, for CUDA,
# the NVIDIA driver plus NVIDIA Container Toolkit.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMAGES_DIR="$SCRIPT_DIR/images"
IMAGE_PREFIX=discamb-test
SUPPORTED_BACKENDS=(cuda)

usage() {
    cat <<EOF
Usage: $(basename "$0") [options] [-- pytest-args...]

Options:
  -s, --systems LIST      comma-separated systems to test (default: every
                          images/<system>-<backend> directory)
  -b, --backends LIST     comma-separated backends to restrict to (default: all
                          detected; supported: ${SUPPORTED_BACKENDS[*]})
  -p, --pydiscamb-dir DIR pyDiSCaMB tree to test (default: $(cd "$SCRIPT_DIR/../../.." && pwd))
  -o, --output-dir DIR    where logs and results go (default: testing/results/<timestamp>)
  -g, --git               test sources cloned from git instead of the local tree;
                          honours PYDISCAMB_REPO, PYDISCAMB_REF, DISCAMB_REPO, DISCAMB_REF
      --rebuild           rebuild images from scratch (--pull --no-cache)
      --no-build          use existing images, fail if missing
  -d, --debug             also show progress, image builds and container setup
                          (always saved under the output directory)
  -h, --help              show this help
EOF
}

DEBUG=0
log() { if [ "$DEBUG" = 1 ]; then printf '==> %s\n' "$*"; fi; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

# Backend plugin convention: detect_<backend> succeeds when that GPU backend is
# usable here; docker_gpu_args_<backend> prints the docker run flags exposing it.

pci_vendor_present() {
    command -v lspci >/dev/null && lspci -nn | grep -Eq "\[(0300|0302|0380)\]:.*\[$1:"
}

cuda_gpu_args=()

# Not `--gpus all`: Docker 29 resolves it through CDI and fails with
# "AMD CDI spec not found" unless an NVIDIA CDI spec exists, so the access
# method the toolkit set up on this host is chosen explicitly.
detect_cuda() {
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        if pci_vendor_present 10de; then
            warn "NVIDIA GPU found on PCI bus but nvidia-smi does not work (driver missing?); skipping cuda"
        fi
        return 1
    fi
    if [ "$DEBUG" = 1 ]; then nvidia-smi -L | sed 's/^/    /'; fi

    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
        cuda_gpu_args=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
    elif docker info --format '{{json .DiscoveredDevices}}' 2>/dev/null | grep -q 'nvidia.com/gpu' \
        || grep -qs 'nvidia.com/gpu' /etc/cdi/* /var/run/cdi/*; then
        cuda_gpu_args=(--device nvidia.com/gpu=all)
    else
        warn "docker has neither the 'nvidia' runtime nor an NVIDIA CDI spec; one-time setup (root):"
        warn "  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
        warn "  or: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        warn "skipping cuda"
        return 1
    fi
    log "cuda: docker GPU access via ${cuda_gpu_args[*]}"
}

docker_gpu_args_cuda() { printf '%s\n' "${cuda_gpu_args[@]}"; }

report_unsupported_gpus() {
    pci_vendor_present 1002 && log "AMD GPU found; no backend images for it yet"
    pci_vendor_present 8086 && log "Intel GPU found; no backend images for it yet"
    return 0
}

systems_arg=""
backends_arg=""
PYDISCAMB_DIR="$SCRIPT_DIR/../../.."
OUTPUT_DIR="$SCRIPT_DIR/results/$(date +%Y%m%d-%H%M%S)"
SOURCE_MODE=local
build_mode=normal
pytest_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--systems) systems_arg=${2:?}; shift 2 ;;
        -b|--backends) backends_arg=${2:?}; shift 2 ;;
        -p|--pydiscamb-dir) PYDISCAMB_DIR=${2:?}; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR=${2:?}; shift 2 ;;
        -g|--git) SOURCE_MODE=git; shift ;;
        --rebuild) build_mode=rebuild; shift ;;
        --no-build) build_mode=none; shift ;;
        -d|--debug) DEBUG=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; pytest_args=("$@"); break ;;
        *) usage >&2; die "unknown option: $1" ;;
    esac
done

command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon"

if [ "$SOURCE_MODE" = local ]; then
    PYDISCAMB_DIR=$(cd "$PYDISCAMB_DIR" && pwd)
    grep -q '^name = "pydiscamb"' "$PYDISCAMB_DIR/pyproject.toml" 2>/dev/null \
        || die "$PYDISCAMB_DIR is not a pyDiSCaMB tree"
fi

log "detecting GPU backends"
backends=()
for b in "${SUPPORTED_BACKENDS[@]}"; do
    if [ -n "$backends_arg" ] && [[ ",$backends_arg," != *",$b,"* ]]; then
        continue
    fi
    if "detect_$b"; then
        log "backend $b: available"
        backends+=("$b")
    fi
done
report_unsupported_gpus
[ ${#backends[@]} -gt 0 ] || die "no usable GPU backend found"

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(cd "$OUTPUT_DIR" && pwd)
running_container=""
trap '[ -n "$running_container" ] && docker kill "$running_container" >/dev/null 2>&1' EXIT
trap 'exit 130' INT TERM

run_logged() {
    local tag=$1 logfile=$2 show=$3 rc
    shift 3
    set +e
    if [ "$show" = 1 ]; then
        "$@" 2>&1 | tee "$logfile" | sed -u "s/^/[$tag] /"
        rc=${PIPESTATUS[0]}
    else
        "$@" >"$logfile" 2>&1
        rc=$?
    fi
    set -e
    return "$rc"
}

build_image() {
    local tag=$1 image=$2 dockerfile=$3 logfile=$4
    case "$build_mode" in
        none)
            docker image inspect "$image" >/dev/null 2>&1 || { echo "image $image missing" >"$logfile"; return 1; }
            ;;
        rebuild)
            run_logged "$tag" "$logfile" "$DEBUG" docker build --pull --no-cache -t "$image" -f "$dockerfile" "$IMAGES_DIR"
            ;;
        *)
            run_logged "$tag" "$logfile" "$DEBUG" docker build -t "$image" -f "$dockerfile" "$IMAGES_DIR"
            ;;
    esac
}

declare -a summary=()
failures=0

for backend in "${backends[@]}"; do
    systems=()
    if [ -n "$systems_arg" ]; then
        IFS=, read -r -a systems <<<"$systems_arg"
    else
        for d in "$IMAGES_DIR"/*-"$backend"/; do
            [ -f "$d/Dockerfile" ] && systems+=("$(basename "$d" "-$backend")")
        done
    fi

    mapfile -t gpu_args < <("docker_gpu_args_$backend")

    for system in "${systems[@]}"; do
        tag="$system-$backend"
        dockerfile="$IMAGES_DIR/$tag/Dockerfile"
        image="$IMAGE_PREFIX:$tag"
        result_dir="$OUTPUT_DIR/$tag"
        mkdir -p "$result_dir"

        if [ ! -f "$dockerfile" ]; then
            warn "no image definition $dockerfile"
            summary+=("$tag: MISSING IMAGE")
            failures=$((failures + 1))
            continue
        fi

        log "[$tag] building image $image"
        if ! build_image "$tag" "$image" "$dockerfile" "$result_dir/image-build.log"; then
            if [ "$DEBUG" != 1 ]; then
                tail -n 30 "$result_dir/image-build.log" | sed "s/^/[$tag] /" >&2
            fi
            summary+=("$tag: IMAGE BUILD FAILED (see $result_dir/image-build.log)")
            failures=$((failures + 1))
            continue
        fi

        run_args=(--rm --init "${gpu_args[@]}"
            -e SOURCE_MODE="$SOURCE_MODE"
            -e PYTEST_ARGS="${pytest_args[*]}"
            -e DEBUG="$DEBUG"
            -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"
            -v "$result_dir:/results")
        if [ "$SOURCE_MODE" = local ]; then
            run_args+=(-v "$PYDISCAMB_DIR:/src:ro")
        else
            for var in PYDISCAMB_REPO PYDISCAMB_REF DISCAMB_REPO DISCAMB_REF; do
                [ -n "${!var:-}" ] && run_args+=(-e "$var=${!var}")
            done
        fi

        running_container="$IMAGE_PREFIX-$tag-$$"
        log "[$tag] running tests"
        start=$SECONDS
        if run_logged "$tag" "$result_dir/run.log" 1 docker run --name "$running_container" "${run_args[@]}" "$image"; then
            summary+=("$tag: PASSED ($((SECONDS - start))s)")
        else
            summary+=("$tag: FAILED ($((SECONDS - start))s, see $result_dir/run.log)")
            failures=$((failures + 1))
        fi
        running_container=""
    done
done

printf '==> summary (results in %s)\n' "$OUTPUT_DIR"
printf '    %s\n' "${summary[@]}"
[ "$failures" -eq 0 ]
