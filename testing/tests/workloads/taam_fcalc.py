"""TAAM structure factor workload, run inside a test environment by the tests.

usage: taam_fcalc.py <structure.pdb> <engine> <d_min>

Prints one ``RESULT <json>`` line with the reflection count, setup and f_calc
wall times in seconds, and this process's peak resident memory in MiB.
"""

import json
import resource
import sys
import time

import pydiscamb
from cctbx import miller
from iotbx import pdb

structure_file, engine, d_min = sys.argv[1], sys.argv[2], float(sys.argv[3])

start = time.perf_counter()
xrs = pdb.input(file_name=structure_file).xray_structure_simple()
xrs.scattering_type_registry(table="xray")
wrapper = pydiscamb.DiscambWrapper(
    xrs,
    method=pydiscamb.FCalcMethod.TAAM,
    engine=engine,
    algorithm="macromol",
    n_cores=1,
)
miller_set = miller.build_set(
    crystal_symmetry=xrs.crystal_symmetry(), anomalous_flag=False, d_min=d_min
)
wrapper.set_indices(miller_set.indices())
setup_s = time.perf_counter() - start

start = time.perf_counter()
f_calc = wrapper.f_calc()
f_calc_s = time.perf_counter() - start

result = {
    "reflections": len(f_calc),
    "setup_s": setup_s,
    "f_calc_s": f_calc_s,
    "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
}
print("RESULT " + json.dumps(result), flush=True)
