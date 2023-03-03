# +
import ase
import torch
from ase.calculators.calculator import Calculator, all_changes
from torch.autograd import grad

from theforce.descriptor.atoms import TorchAtoms


def partial(model, atoms, indices=None):
    if indices is None:
        indices = range(0, atoms.natoms)
    locs = [atoms.local(i) for i in indices]
    # energy
    energy = model([locs], "energy", enable_grad=True)
    # forces
    rgrad = grad(energy, atoms.xyz, retain_graph=True, allow_unused=True)[0]
    forces = torch.zeros_like(atoms.xyz) if rgrad is None else -rgrad
    # cellgrad
    (cellgrad,) = grad(energy, atoms.lll, allow_unused=True)
    if cellgrad is None:
        cellgrad = torch.zeros_like(atoms.lll)
    return (x.detach().numpy() for x in (energy.view([]), forces, cellgrad))


class BaseCalculator(Calculator):
    implemented_properties = ["energy", "forces", "free_energy", "stress"]

    def __init__(self, potential, calculate=partial, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.potential = potential
        self.ext_calculate = calculate

    def calculate(self, _atoms=None, properties=["energy"], system_changes=all_changes):
        # ase to torch
        if type(_atoms) == ase.atoms.Atoms:
            atoms = TorchAtoms(ase_atoms=_atoms, cutoff=self.potential._cutoff)
        else:
            atoms = _atoms
        Calculator.calculate(self, atoms, properties, system_changes)
        # update without descriptors
        self.atoms.update(
            posgrad=True,
            cellgrad=True,
            forced=True,
            build_locals=False,
            descriptors=self.potential.gp.kern.kernels,
        )
        # calculate
        energy, forces, cellgrad = self.ext_calculate(self.potential, self.atoms)
        # stress
        stress1 = -(forces[:, None] * atoms.positions[..., None]).sum(axis=0)
        stress2 = (cellgrad[:, None] * atoms.cell[..., None]).sum(axis=0)
        try:
            volume = atoms.get_volume()
        except ValueError:
            volume = -2.0  # here stress2=0, thus trace(stress) = virial (?)
        stress = (stress1 + stress2) / volume
        # results
        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["free_energy"] = self.results["energy"]
        self.results["stress"] = stress.flat[[0, 4, 8, 5, 2, 1]]
