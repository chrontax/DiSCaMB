"""TAAM structure factors and target derivatives for one engine, saved for comparison.

usage: taam_values.py <structure.pdb> <engine> <d_min> <out.npz>

Saves f_calc (complex, per reflection) and the per-atom derivatives site (n x 3),
adp (all atoms' ADP derivatives concatenated: 1 per isotropic, 6 per anisotropic
atom) and occupancy (n). d_target/d_f_calc is a seeded pseudo-random complex
vector, so every engine gets identical input and the derivatives are comparable.
"""

import sys

import numpy as np
from cctbx import miller
from iotbx import pdb

import pydiscamb

structure_file, engine, d_min, out_file = (
    sys.argv[1],
    sys.argv[2],
    float(sys.argv[3]),
    sys.argv[4],
)

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

f_calc = np.array(list(wrapper.f_calc()), dtype=complex)
rng = np.random.default_rng(0)
d_target_d_f_calc = rng.standard_normal(f_calc.size) + 1j * rng.standard_normal(
    f_calc.size
)
derivatives = wrapper.d_target_d_params([complex(x) for x in d_target_d_f_calc])

np.savez(
    out_file,
    f_calc=f_calc,
    site=np.array([d.site_derivatives for d in derivatives], dtype=float),  # 2d
    adp=np.concatenate(
        [np.asarray(d.adp_derivatives, dtype=float) for d in derivatives]
    ),
    occupancy=np.array([d.occupancy_derivatives for d in derivatives], dtype=float),
)
