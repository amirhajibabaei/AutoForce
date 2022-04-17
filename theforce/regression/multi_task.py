# +
from theforce.regression.gppotential import PosteriorPotential
import theforce.distributed as distrib
import torch


class MultiTaskPotential(PosteriorPotential):
    """
    Same as PosteriorPotential, except multiple potential energies
    are used simultaneouly for regression using a multi-task-learning
    algorithm.
    
    ad hoc parameters:
        scaled_noise (sigma) is fixed at 0.01.
        A ridge of 0.01 is added to the tasks_kern.
        
    TODO:
        tasks_kern should be obtained by optimization.    
        
    """

    def __init__(self, tasks, *args, **kwargs):
        """
        tasks: number of potential energy types.
        
        """
        super().__init__(*args, **kwargs)
        self.tasks = tasks

    def make_munu(self, *args, **kwargs):

        # *** targets ***
        energies, forces = [], []
        atom_types = set()
        atom_counts = torch.zeros(len(self.data), 119)
        for i, atoms in enumerate(self.data):
            for z, c in atoms.counts().items():
                atom_types.add(z)
                atom_counts[i, z] = c
            energies.append(atoms.target_energy.view(-1))
            forces.append(atoms.target_forces.view(-1))
        energies = torch.cat(energies)
        forces = torch.cat(forces)
        targets = torch.cat([energies, forces])
        size = targets.numel()
        atom_types = sorted(atom_types)

        # *** kernel matrices ***
        # main block:
        kern_1 = torch.cat([self.Ke, self.Kf])

        # Constant-energy-shift block:
        ntypes = len(atom_types)
        ke_shift = atom_counts[:, atom_types]
        kf_shift = torch.zeros(self.Kf.size(0), ntypes)
        kern_2 = torch.cat([ke_shift, kf_shift])

        # patch all blocks:
        kern = torch.cat([kern_1, kern_2], dim=1)

        # *** legacy stuff ***
        sigma = 0.01
        self.scaled_noise = {'all': sigma}
        chol = torch.linalg.cholesky(self.M)
        self.ridge = torch.tensor(0.)
        self.choli = chol.inverse().contiguous()
        self.Mi = self.choli.t() @ self.choli
        chol_size = chol.size(0)
        self.mean._weight = {}
        self.mean.weights = {}

        # *** sgpr extensions ***
        sgpr = True
        if sgpr:
            _kern_1 = sigma * chol.t()
            _kern_2 = torch.zeros(chol_size, ntypes)
            _kern = torch.cat([_kern_1, _kern_2], dim=1)
            kern = torch.cat([kern, _kern])
            _targets = torch.zeros(self.tasks * chol_size)
            targets = torch.cat([targets, _targets])

        # *** tasks kernel ***
        # covariance between tasks:
        # in principle this should be optimized as a hyper parameter.
        # for now we set it equal to correlation of forces in different
        # tasks.
        # for independent tasks, set:
        # self.tasks_kern = torch.eye(self.tasks)
        self.tasks_kern = tasks_correlation(forces.view(-1, self.tasks))
        # a small ridge maybe needed in case tasks are identical:
        self.tasks_kern += torch.eye(self.tasks) * 1e-2

        # *** solution ***
        design = torch.kron(kern, self.tasks_kern)
        solution, predictions = least_squares(design, targets)
        self.multi_mu = solution
        self.multi_types = {z: i for i, z in enumerate(atom_types)}

        # *** stats ***
        # TODO: per-task vscales?
        split = chol_size * self.tasks
        self.mu = solution[:split].reshape(-1, self.tasks)
        self.make_stats((targets[:size], predictions[:size]))

    def predict_multitask_energies(self, kern, local_numbers):
        assert kern.size(0) == len(local_numbers)
        ntypes = len(self.multi_types)
        kern_shift = torch.zeros(kern.size(0), ntypes)
        for i, z in enumerate(local_numbers):
            if z in self.multi_types:
                kern_shift[i, self.multi_types[z]] = 1.
            else:
                pass
                # raise RuntimeError(f'unseen atomic number {z}')
        kern = torch.cat([kern, kern_shift], dim=1)
        kern = torch.kron(kern, self.tasks_kern)
        energies = (kern @ self.multi_mu).reshape(-1, self.tasks).sum(dim=0)
        return [e for e in energies]


def least_squares(design, targets, trials=10):
    """
    Minimizes 
        || design @ solution - targets ||
    using torch.linalg.lstsq.
    
    Returns
        solution, predictions (= design @ solution)
        
    Since torch.linalg.lstsq is non-deteministic,
    the best solution of "trials" is return.
    """
    best = None
    predictions = None
    for _ in range(trials):
        sol = torch.linalg.lstsq(design, targets)
        pred = design @ sol.solution
        err = (pred - targets).abs().mean()
        if not best or err < best:
            solution = sol.solution
            predictions = pred
            best = err
    return solution, predictions


def tasks_correlation(f):
    a = f.t() @ f
    b = a.diag().sqrt()[None]
    return a / (b * b.t())
