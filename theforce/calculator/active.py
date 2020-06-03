import numpy as np
import torch
from torch.autograd import grad
from theforce.util.util import date
from theforce.descriptor.atoms import TorchAtoms, AtomsData, LocalsData
from theforce.regression.gppotential import PosteriorPotential, PosteriorPotentialFromFolder
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
import ase
import warnings


class Tweak:

    def __init__(self, ediff=0.1, fdiff=0.1, beta=0.1):
        self.ediff = ediff
        self.fdiff = fdiff
        self.beta = beta
        self.skip_after_fp = 3


class ActiveCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, calculator, model, tweak=Tweak(), process_group=None,
                 logfile='accalc.log', verbose=0, **kwargs):
        Calculator.__init__(self, **kwargs)
        self._calc = calculator
        self.get_model(model)
        self.tweak = tweak
        self.process_group = process_group
        self.verbose = verbose
        self.logfile = logfile

        self.step = 0
        self.log('active calculator says Hello!', mode='w')
        self.model._cutoff = max([d.descriptor._radial.rc
                                  for d in self.model.descriptors])
        self.skip = 0

    def get_model(self, model):
        if type(model) == str:
            self.model = PosteriorPotentialFromFolder(model)
        elif type(model) == PosteriorPotential:
            self.model = model
        else:
            self.model = PosteriorPotential(model)

    @property
    def size(self):
        return self.model.ndata, len(self.model.X)

    def calculate(self, _atoms=None, properties=['energy'], system_changes=all_changes):
        if type(_atoms) == ase.atoms.Atoms:
            atoms = TorchAtoms(ase_atoms=_atoms)
            uargs = {'cutoff': self.model._cutoff,
                     'descriptors': self.model.gp.kern.kernels}
            self.to_ase = True
        else:
            atoms = _atoms
            uargs = {}
            self.to_ase = False
        if _atoms is not None and self.process_group is not None:
            atoms.attach_process_group(self.process_group)
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.update(posgrad=True, cellgrad=True,
                          forced=True, dont_save_grads=True, **uargs)
        # build a model
        if self.step == 0:
            if self.model.ndata == 0:
                self.initiate_model()
            self.log('size: {} {}'.format(*self.size))
        # kernel
        a = self.model.gp.kern(self.atoms, self.model.X)
        # energy
        energy = (a@self.model.mu).sum()
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(enegy)
        # forces
        rgrad = grad(energy, self.atoms.xyz, retain_graph=True,
                     allow_unused=True)[0]
        forces = torch.zeros_like(self.atoms.xyz) if rgrad is None else -rgrad
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(forces)
        # stress
        stress1 = -(forces[:, None]*self.atoms.xyz[..., None]).sum(dim=0)
        cellgrad, = grad(energy, self.atoms.lll, allow_unused=True)
        if cellgrad is None:
            cellgrad = torch.zeros_like(self.atoms.lll)
        if self.atoms.is_distributed:
            torch.distributed.all_reduce(cellgrad)
        stress2 = (cellgrad[:, None]*self.atoms.lll[..., None]).sum(dim=0)
        try:
            volume = self.atoms.get_volume()
        except ValueError:
            volume = -2  # here stress2=0, thus trace(stress) = virial (?)
        stress = (stress1 + stress2).detach().numpy() / volume
        # results
        self.results['energy'] = energy.detach().numpy()
        self.results['forces'] = forces.detach().numpy()
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        self.log('{} {}'.format(float(energy), self.atoms.get_temperature()))
        #
        if self.skip > 0:
            self.skip -= 1
        else:
            self.update(a)
        #
        self.step += 1

    def initiate_model(self):
        atoms = self.snapshot()
        i = atoms.first_of_each_atom_type()
        locs = atoms.gathered()
        inducing = LocalsData([locs[j] for j in i])
        data = AtomsData([atoms])
        self.model.set_data(data, inducing)
        for j in range(atoms.natoms):
            if j not in i:
                self.model.add_1inducing(locs[j], self.tweak.ediff)
        self.skip += 1 + self.tweak.skip_after_fp

    def _exact(self, copy):
        tmp = copy.as_ase() if self.to_ase else copy
        tmp.set_calculator(self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        if self.rank == 0:
            ase.io.Trajectory('_calc.traj', 'a').write(tmp)
        self.log('exact energy: {}'.format(energy))
        #
        if self.model.ndata > 0:
            dE = self.results['energy'] - energy
            df = abs(self.results['forces'] - forces)
            self.log(
                f'errors:  dE: {dE}  df_max: {df.max()}  df_mean: {df.mean()}')
        return energy, forces

    def snapshot(self, fake=False, copy=None):
        if copy is None:
            copy = self.atoms.copy()
        if fake:
            energy = self.results['energy']
            forces = self.results['forces']
        else:
            energy, forces = self._exact(copy)
        copy.set_calculator(SinglePointCalculator(copy, energy=energy,
                                                  forces=forces))
        copy.set_targets()
        return copy

    def head(self, energy_and_forces=None):
        added = self.model.data[-1]
        if energy_and_forces is None:
            energy, forces = self._exact(added)
        added.calc.results['energy'] = energy
        added.calc.results['forces'] = forces
        added.set_targets()
        self.model.make_munu()

    def gather(self, x):
        _x = torch.zeros(self.atoms.natoms)
        _x[self.atoms.indices] = beta
        return torch.distributed.all_reduce(_x)

    def update_inducing(self, a):
        b = self.model.choli@a.T
        alpha = torch.cat([self.model.gp.kern(x, x)
                           for x in self.atoms]).view(-1)
        beta = (1 - (b.T@b).diag()/alpha).clamp(min=0.).sqrt()
        if self.atoms.is_distributed:
            alpha = self.gather(alpha)
            beta = self.gather(beta)
        q = torch.argsort(beta, descending=True)
        added_beta = 0
        added_diff = 0
        for k in q:
            loc = self.atoms.local(k)
            if loc.number in self.model.gp.species:
                if beta[k] > self.tweak.beta:
                    self.model.add_inducing(loc)
                    added_beta += 1
                else:
                    _ediff = (self.tweak.ediff if len(self.model.X) > 1
                              else torch.finfo().tiny)
                    delta = self.model.add_1inducing(
                        loc, _ediff, detach=False)
                    if delta >= self.tweak.ediff:
                        added_diff += 1
                    else:
                        break
        added = added_beta + added_diff
        if added > 0:
            self.log('added indu: {} ({},{})-> size: {} {}'.format(
                added, added_beta, added_diff, *self.size))
        return added

    def update_data(self, try_fake=True):
        n = self.model.ndata
        new = self.snapshot(fake=try_fake)
        self.model.add_1atoms(new, self.tweak.ediff, self.tweak.fdiff)
        if try_fake:
            self.head()
        added = self.model.ndata - n
        if added > 0:
            self.log('added data: {} -> size: {} {}'.format(
                added, *self.size))
        return added

    def update(self, a):
        if self.model.ndata < 3:
            n = self.update_data(False)
            m = self.update_inducing(a)
        else:
            m = self.update_inducing(a)
            n = self.update_data(True) if m > 0 else 0
        if n > 0:
            self.skip += self.tweak.skip_after_fp

    @property
    def rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0

    def log(self, mssge, mode='a'):
        if self.rank == 0:
            with open(self.logfile, mode) as f:
                f.write('{} {} {}\n'.format(date(), self.step, mssge))
                if self.verbose:
                    print('{} {} {}'.format(date(), self.step, mssge))


def parse_logfile(file='accalc.log'):
    energies = []
    temperatures = []
    exact_energies = []
    errors = []
    for line in open(file):
        split = line.split()[2:]

        try:
            step = int(split[0])
        except IndexError:
            continue

        try:
            energies += [(step, float(split[1]))]
            temperatures += [(step, float(split[2]))]
        except:
            pass

        if 'exact energy' in line:
            exact_energies += [(step, float(split[3]))]

        if 'errors' in line:
            errors += [(step, [float(v) for v in split[3:8:2]])]
    return energies, exact_energies, errors
