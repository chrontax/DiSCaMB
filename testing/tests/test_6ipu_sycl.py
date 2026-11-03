import json
from pathlib import Path

from tools import GpuMemoryProfiler

WORKLOAD = Path(__file__).parent / "workloads" / "taam_fcalc.py"
STRUCTURE = "scripts/examples/6IPU/6IPU.pdb"
ENGINE = "CPU_v4"
D_MIN = 2.0


def _workload_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    raise AssertionError(f"workload printed no RESULT line:\n{stdout}")


def test_6ipu_sycl_resources(run_in_env, source_dir, results_dir):
    profiler = GpuMemoryProfiler()
    profiler.start()
    run = run_in_env("sycl", WORKLOAD, args=[str(source_dir / STRUCTURE), ENGINE, str(D_MIN)])
    gpu = profiler.stop()
    workload = _workload_result(run.stdout)

    report = {
        "structure": STRUCTURE,
        "env": "sycl",
        "engine": ENGINE,
        "d_min": D_MIN,
        "reflections": workload["reflections"],
        "setup_s": workload["setup_s"],
        "f_calc_s": workload["f_calc_s"],
        "wall_s": run.wall_time_s,
        "host_peak_rss_mib": workload["peak_rss_mib"],
        "gpu_peak_mib": gpu.peak_mib,
        "gpu_mean_mib": gpu.mean_mib,
        "gpu_baseline_mib": gpu.baseline_mib,
        "gpu_peak_above_baseline_mib": gpu.peak_above_baseline_mib,
        "gpu_samples": gpu.samples,
    }
    (results_dir / "6IPU-sycl.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    assert workload["reflections"] > 0
    assert gpu.peak_above_baseline_mib > 0, "GPU memory never rose: Engine4 did not run on the GPU"
