# +
import theforce.distributed as dist
from theforce.util.parallel import mpi_init
from theforce.cl.mlmd import mlmd, read_md
from theforce.cl import ARGS
from ase.io import read
import os


group = ARGS['process_group']
os.environ['CORES_FOR_ML'] = str(dist.get_world_size())
try:
    from theforce.calculator import vasp
    calc_script = vasp.__file__
except:
    raise

atoms = read('POSCAR')
mlmd(atoms, calc_script=calc_script, **read_md('MD'), group=group)
