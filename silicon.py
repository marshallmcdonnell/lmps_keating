
#--------------------------------------------------#
# ASE - Build system and output lammps-data file
#--------------------------------------------------#

from __future__ import print_function

from ase.calculators.lammpslib import LAMMPSlib, write_lammps_data
from ase.lattice.cubic import Diamond
from ase.visualize import view


header = ['units real', 
      'atom_style molecular',
      'pair_style zero 10.0',
      'bond_style harmonic',
      'angle_style harmonic']

cmds = ["pair_coeff * * ",
    "bond_coeff * 100.0 2.351",
    "angle_coeff * 100.0 109.47", 
    "fix 1 all nve"]

lat_dims=(2,2,2)
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
               log_file='test.log', keep_alive=True)

atoms.set_calculator(calc)

write_lammps_data('silicon.data', 
              atoms, 
              {'Si': 1}, 
              cutoff=2.351,
              bond_types=[(14,14)],
              angle_types=[(14,14,14)],
              units='real')

#--------------------------------------------------#
# PyLammps
#--------------------------------------------------#

from lammps import IPyLammps

L = IPyLammps()
L.units('real')
L.dimension(3)
L.boundary('p p p')
L.atom_style('full')
L.read_data("./silicon.data")
L.pair_style('zero', 10.0)
L.bond_style('harmonic')
L.angle_style('harmonic')

L.pair_coeff('*', '*')
L.bond_coeff('*',100.0,2.351)
L.angle_coeff('*',100.0,109.47)

L.neighbor(5.0,'bin')
L.fix('1 all nve')
L.dump('1 all atom 100 dump.silicon.lammpstrj')

L.velocity('all create 300.0 2341')

L.command('thermo_style custom step time etotal pe ke')
L.thermo(100)

L.command('run 5000')
#MPI.Finalize()
