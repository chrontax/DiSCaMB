# discamb testing engine

End-to-end tests for the SYCL (GPU) build of discamb, run through pyDiSCaMB in
docker containers.

## Running (Linux)

```bash
lib/discamb/testing/test-linux.sh                 # every detected backend x every system
lib/discamb/testing/test-linux.sh -s ubuntu       # one system
lib/discamb/testing/test-linux.sh -- -k memory    # extra pytest arguments
lib/discamb/testing/test-linux.sh --git           # test git sources instead of the local tree
```

Host requirements: docker, plus for the `cuda` backend the NVIDIA driver and
NVIDIA Container Toolkit registered with docker, either as the `nvidia` runtime
(`sudo nvidia-ctk runtime configure --runtime=docker`, restart docker) or as a
CDI spec (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`). The
script uses whichever is present; it does not use `--gpus all`, which fails on
Docker 29 without a CDI spec. The script installs nothing on the host.

The sycl build enables `TAAM_WITH_GPU_IMP_26` so `HansenCoppens_SF_Engine4`
(engine `CPU_v4`) is compiled. Logs, build
logs and `junit.xml` land in `testing/results/<timestamp>/<system>-<backend>/`.

`--git` clones `PYDISCAMB_REPO@PYDISCAMB_REF` (default discamb-project/pyDiSCaMB
`main`) and swaps the `lib/discamb` submodule for `DISCAMB_REPO@DISCAMB_REF`
(default chrontax/DiSCaMB `main`).

## Layout

- `test-linux.sh` detects GPU backends and runs one container per
  `images/<system>-<backend>` image.
- `images/<system>-<backend>/Dockerfile` holds the system packages and CUDA base
  image. Build args: `CUDA_VERSION` (default 12.9.2), `DPCPP_VERSION` (default v6.3.0).
- `images/common/provision.sh` installs micromamba, the DPC++ release and the
  `discamb-cpu` / `discamb-sycl` environments (`environment.yml`) at image build time.
- `images/common/run-tests.sh` is the container entrypoint. It builds pyDiSCaMB
  into both environments (CPU with the system compiler, SYCL with DPC++),
  checks that the SYCL runtime sees a backend device, then runs pytest over
  `tests/` in the sycl environment.
- `images/common/env-exec <cpu|sycl> cmd...` runs a command in one environment.
- `tests/conftest.py` provides fixtures for tests: `run_in_env(env, script)`
  runs Python in the `cpu` or `sycl` environment (timed); also `backend`,
  `system`, `results_dir`, `source_dir`.
- `tests/tools.py` holds diagnostics importable from tests. `GpuMemoryProfiler`
  samples GPU memory every 10 ms with the vendor tool of the detected backend
  (`cuda` -> `nvidia-smi --query-gpu=memory.used --loop-ms=10`):

  ```python
  from tools import GpuMemoryProfiler

  profiler = GpuMemoryProfiler()
  profiler.start()
  run_in_env("sycl", script)
  profile = profiler.stop()
  assert profile.peak_above_baseline_mib < 2048  # also .peak_mib, .mean_mib
  ```

  The backend comes from `DISCAMB_TEST_BACKEND`, falling back to whichever
  vendor tool is on `PATH`. Sampling covers the whole device, so it also
  measures work done by a child process started through `run_in_env`.
  `start()` takes a one-shot reading as `baseline_mib` and raises there if the
  tool does not work; `stop()` raises if the profiled window was too short for
  the sampler to produce a reading (vendor tools need ~0.1 s to start).

## Adding a backend

Add `detect_<backend>` and `docker_gpu_args_<backend>` to `test-linux.sh`,
append the backend to `SUPPORTED_BACKENDS`, and add
`images/<system>-<backend>/Dockerfile` setting `TEST_BACKEND` (the SYCL
`ONEAPI_DEVICE_SELECTOR` backend name) and `SYCL_TARGETS`. For memory
profiling, add a `_MemorySampler` subclass to `tests/tools.py` (backend name,
vendor tool, command, line parser) and list it in `SAMPLERS`.
