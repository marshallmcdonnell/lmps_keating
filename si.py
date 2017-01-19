"""Get energy from a LAMMPS calculation"""

from __future__ import print_function

from ase.calculators.lammpslib import LAMMPSlib, write_lammps_data
from ase.lattice.cubic import Diamond

if __name__ == "__main__":

    header = ['units real', 
              'atom_style molecular',
              'pair_style zero 10.0',
              'bond_style harmonic',
              'angle_style harmonic']

    cmds = ["pair_coeff * * ",
            "bond_coeff * 100.0 2.351",
            "angle_coeff * 100.0 109.47", 
            "fix 1 all nve"]

    lat_dims=(1,1,1)
    atoms = Diamond( symbol='Si', size=lat_dims )
    print(atoms.get_positions())

    # parallel via mpi4py
    try:
        from mpi4py import MPI
        me = MPI.COMM_WORLD.Get_rank()
        nprocs = MPI.COMM_WORLD.Get_size()
    except:
        me = 0

    # Set LAMMPS as Calculator
    calc = LAMMPSlib(lmpcmds=cmds,
                       lammps_header=header,
                       atom_types={'Si': 1},
                       read_molecular_info=True,
                       log_file='test.log', keep_alive=True)

    atoms.set_calculator(calc)

    write_lammps_data('si.data', 
                      atoms, 
                      {'Si': 1}, 
                      cutoff=2.351,
                      bond_types=[(14,14)],
                      angle_types=[(14,14,14)],
                      units='real')


