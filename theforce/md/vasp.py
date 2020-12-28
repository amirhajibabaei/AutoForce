# +
import torch.distributed as dist
from theforce.util.parallel import mpi_init
from theforce.md.mlmd import mlmd, read_md
from ase.io import read
import os


group = mpi_init()
os.environ['CORES_FOR_ML'] = str(dist.get_world_size())
try:
    from theforce.calculator import vasp
    calc_script = vasp.__file__
except:
    raise

atoms = read('POSCAR')
mlmd(atoms, calc_script, **read_md('MD'), group=group)
