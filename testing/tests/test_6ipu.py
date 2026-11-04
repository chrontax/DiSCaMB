import json
from pathlib import Path

import numpy as np
import pytest
from tools import GpuMemoryProfiler

WORKLOAD = Path(__file__).parent / "workloads" / "taam_fcalc.py"
QUANTITIES = ("f_calc", "site", "adp", "occupancy")
STRUCTURE = "scripts/examples/6IPU/6IPU.pdb"
D_MIN = 2.0


def _run_workload(run_in_env, source_dir, results_dir, backend, build, engine):
    values_file = results_dir / f"6IPU-values-{engine}.npz"
    run = run_in_env(
        WORKLOAD,
        args=[str(source_dir / STRUCTURE), engine, str(D_MIN), str(values_file)],
    )
    for line in reversed(run.stdout.splitlines()):
        if line.startswith("RESULT "):
            workload = json.loads(line[len("RESULT ") :])
            break
    else:
        raise AssertionError(f"workload printed no RESULT line:\n{run.stdout}")
    report = {
        "structure": STRUCTURE,
        "backend": backend,
        "build": build,
        "engine": engine,
        "d_min": D_MIN,
        "reflections": workload["reflections"],
        "setup_s": workload["setup_s"],
        "f_calc_s": workload["f_calc_s"],
        "derivatives_s": workload["derivatives_s"],
        "wall_s": run.wall_time_s,
        "host_peak_rss_mib": workload["peak_rss_mib"],
    }
    with np.load(values_file) as data:
        values = {name: data[name] for name in QUANTITIES}
    return report, values


def _save(report, results_dir):
    (results_dir / f"6IPU-{report['engine']}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


@pytest.fixture(scope="module")
def cpu_run(run_in_env, source_dir, results_dir, backend, build):
    return _run_workload(run_in_env, source_dir, results_dir, backend, build, "CPU")


@pytest.fixture(scope="module")
def sycl_run(sycl_warmup, run_in_env, source_dir, results_dir, backend, build):
    profiler = GpuMemoryProfiler()
    profiler.start()
    report, values = _run_workload(
        run_in_env, source_dir, results_dir, backend, build, "CPU_v4"
    )
    gpu = profiler.stop()
    report.update(
        gpu_peak_mib=gpu.peak_mib,
        gpu_mean_mib=gpu.mean_mib,
        gpu_baseline_mib=gpu.baseline_mib,
        gpu_peak_above_baseline_mib=gpu.peak_above_baseline_mib,
        gpu_samples=gpu.samples,
    )
    return report, values


@pytest.mark.sycl
def test_6ipu_sycl_resources(sycl_run, results_dir):
    report, _ = sycl_run
    _save(report, results_dir)

    assert report["reflections"] > 0
    assert report["gpu_peak_above_baseline_mib"] > 0, (
        "GPU memory never rose: Engine4 did not run on the GPU"
    )


def test_6ipu_cpu_resources(cpu_run, results_dir):
    report, _ = cpu_run
    _save(report, results_dir)

    assert report["reflections"] > 0


@pytest.mark.sycl
def test_6ipu_sycl_matches_cpu(cpu_run, sycl_run, results_dir):
    _, cpu = cpu_run
    _, sycl = sycl_run

    report = {
        "structure": STRUCTURE,
        "d_min": D_MIN,
        "reference": "CPU",
        "compared": "CPU_v4",
    }
    for name in QUANTITIES:
        assert cpu[name].shape == sycl[name].shape, (
            f"{name}: {cpu[name].shape} vs {sycl[name].shape}"
        )
        diff = np.abs(sycl[name] - cpu[name]).ravel()
        assert np.isfinite(diff).all(), f"{name}: non-finite values"
        report[name] = {
            "values": int(diff.size),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(np.median(diff)),
        }
    (results_dir / "6IPU-compare.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
