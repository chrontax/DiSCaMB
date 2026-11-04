"""TAAM structure factor workload, run inside a test environment by the tests.

usage: taam_fcalc.py <structure.pdb> <engine> <d_min> [<out.npz>]

Prints one ``RESULT <json>`` line with the reflection count, setup and f_calc
wall times in seconds, and this process's peak resident memory in MiB measured
right after f_calc.

With ``out.npz`` it then also computes target derivatives and saves f_calc
(complex, per reflection) and the per-atom derivatives site (n x 3), adp (all
atoms' ADP derivatives concatenated: 1 per isotropic, 6 per anisotropic atom)
and occupancy (n); RESULT then also carries ``derivatives_s``.
d_target/d_f_calc is a seeded pseudo-random complex vector, so every engine
gets identical input and the derivatives are comparable.
"""

import json
import resource
import sys
import time

import numpy as np
import pydiscamb
from cctbx import miller
from iotbx import pdb

structure_file, engine, d_min = sys.argv[1], sys.argv[2], float(sys.argv[3])
out_file = sys.argv[4] if len(sys.argv) > 4 else None

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

if out_file is not None:
    f_calc = np.array(list(f_calc), dtype=complex)
    rng = np.random.default_rng(0)
    d_target_d_f_calc = rng.standard_normal(f_calc.size) + 1j * rng.standard_normal(
        f_calc.size
    )
    start = time.perf_counter()
    derivatives = wrapper.d_target_d_params([complex(x) for x in d_target_d_f_calc])
    result["derivatives_s"] = time.perf_counter() - start
    np.savez(
        out_file,
        f_calc=f_calc,
        site=np.array([d.site_derivatives for d in derivatives], dtype=float),  # 2d
        adp=np.concatenate(
            [np.asarray(d.adp_derivatives, dtype=float) for d in derivatives]
        ),
        occupancy=np.array(
            [d.occupancy_derivatives for d in derivatives], dtype=float
        ),
    )

print("RESULT " + json.dumps(result), flush=True)
