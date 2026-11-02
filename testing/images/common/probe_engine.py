"""Startup probe: computes TAAM structure factors for a small structure with the
given Hansen-Coppens engine and exits non-zero if that engine is unavailable.

usage: probe_engine.py <structure.pdb> <engine>   (engine: CPU, CPU_v4, ...)
"""

import sys

import pydiscamb
from iotbx import pdb

structure_file, engine = sys.argv[1], sys.argv[2]
xrs = pdb.input(file_name=structure_file).xray_structure_simple()
xrs.scattering_type_registry(table="xray")
wrapper = pydiscamb.DiscambWrapper(
    xrs,
    method=pydiscamb.FCalcMethod.TAAM,
    engine=engine,
    algorithm="macromol",
    n_cores=1,
)
f_calc = wrapper.f_calc(2.0)
if len(f_calc) == 0:
    sys.exit(f"engine {engine}: no structure factors computed")
print(f"engine {engine}: {len(f_calc)} structure factors")
