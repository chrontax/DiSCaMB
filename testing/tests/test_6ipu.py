import json
from pathlib import Path

import pytest

from tools import GpuMemoryProfiler

WORKLOAD = Path(__file__).parent / "workloads" / "taam_fcalc.py"
STRUCTURE = "scripts/examples/6IPU/6IPU.pdb"
D_MIN = 2.0


def _run_workload(run_in_env, source_dir, backend, build, engine):
    run = run_in_env(WORKLOAD, args=[str(source_dir / STRUCTURE), engine, str(D_MIN)])
    for line in reversed(run.stdout.splitlines()):
        if line.startswith("RESULT "):
            workload = json.loads(line[len("RESULT "):])
            break
    else:
        raise AssertionError(f"workload printed no RESULT line:\n{run.stdout}")
    return {
        "structure": STRUCTURE,
        "backend": backend,
        "build": build,
        "engine": engine,
        "d_min": D_MIN,
        "reflections": workload["reflections"],
        "setup_s": workload["setup_s"],
        "f_calc_s": workload["f_calc_s"],
        "wall_s": run.wall_time_s,
        "host_peak_rss_mib": workload["peak_rss_mib"],
    }


def _save(report, results_dir):
    (results_dir / f"6IPU-{report['engine']}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


@pytest.mark.sycl
@pytest.mark.usefixtures("sycl_warmup")
def test_6ipu_sycl_resources(run_in_env, source_dir, results_dir, backend, build):
    profiler = GpuMemoryProfiler()
    profiler.start()
    report = _run_workload(run_in_env, source_dir, backend, build, "CPU_v4")
    gpu = profiler.stop()
    report.update(
        gpu_peak_mib=gpu.peak_mib,
        gpu_mean_mib=gpu.mean_mib,
        gpu_baseline_mib=gpu.baseline_mib,
        gpu_peak_above_baseline_mib=gpu.peak_above_baseline_mib,
        gpu_samples=gpu.samples,
    )
    _save(report, results_dir)

    assert report["reflections"] > 0
    assert gpu.peak_above_baseline_mib > 0, "GPU memory never rose: Engine4 did not run on the GPU"


def test_6ipu_cpu_resources(run_in_env, source_dir, results_dir, backend, build):
    report = _run_workload(run_in_env, source_dir, backend, build, "CPU")
    _save(report, results_dir)

    assert report["reflections"] > 0
