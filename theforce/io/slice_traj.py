# +
import os
import sys

from ase.io import Trajectory, read


def reduce_traj(traj, x):
    if traj.endswith(".traj"):
        if os.path.isfile(traj):
            reduced = traj.replace(".traj", f"_r{x}.traj")
            new = Trajectory(reduced, "w")
            for atoms in read(traj, f"::{x}"):
                new.write(atoms)
            new.close()
            assert os.system(f"rm -f {traj}") == 0
            print(f"{traj} -> {reduced}")
        else:
            print(f"{traj} not found")
    else:
        print(f"{traj} not a traj file")


if __name__ == "__main__":
    x = int(sys.argv[1])
    for k, traj in enumerate(sys.argv[2:]):
        reduce_traj(traj, x)
