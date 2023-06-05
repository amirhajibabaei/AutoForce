# +
import numpy as np
import torch
from scipy.optimize import minimize
import math

import theforce.distributed as distrib
from theforce.regression.gppotential import PosteriorPotential

from theforce.regression.algebra import jitcholesky


class MultiTaskPotential(PosteriorPotential):
    """
    Same as PosteriorPotential, except multiple potential energies
    are used simultaneouly for regression using a multi-task-learning
    algorithm.

    ad hoc parameters:
        scaled_noise (sigma) is fixed at 0.01.
        A ridge of 0.01 is added to the tasks_kern.

    tasks: The number of potential energy surface to learn
    tasks_kern_optimization: Flag for tasks_kern optimization 
    niter_tasks: The number of tasks kernel optimization iteration
    algo: torch linear algebra algorithm
    tasks_kern_L: Lower triangle kernel matrix
    sigma_reg: Sigma regularization scheme
        - None:  Constant regularization 
        - am: Adaptive regularization over multi_mu 
        - default: Default single task sigma

    Energy shift: We shift the average energy average for better training and 
                  for transferrability of the model to extended systems 
        - None: no energy shift
        - opt: constant energy shift term per element is determined from a SGPR equation. 
        - opt-single: the same constant energy shift for all elements    
        - pre: Use a pre-calculated isolated atom energy of each element type
        - preopt: Based on an atom energy, optimize the shift level 
    """

    def __init__(
        self, 
        tasks, 
        tasks_kern_optimization, 
        niter_tasks, 
        algo, 
        sigma_reg=None, 
        alpha_reg=0.001, 
        shift='opt-single', 
        *args, **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.tasks_kern_L = torch.eye(self.tasks)
        self.tasks_kern_L += 1e-2
        self.tasks_kern = torch.eye(self.tasks)
        self.tasks_kern_optimization = tasks_kern_optimization
        self.niter_tasks = niter_tasks
        self.algo = algo
        self.multi_mu = None
        self.sigma_reg = sigma_reg
        self.alpha_reg = alpha_reg
        self.shift = shift
        self.pre_energy_shift={0: 0.0, 1: -13.6766048214, 8: -429.072746017}

    def make_munu(self, *args, **kwargs):

        # *** targets ***
        energies, forces = [], []
        atom_types = set()

        # *** atom counts ***
        # atom type 120 for dummy element Z - might change to default 119 in future
        atom_counts = torch.zeros(len(self.data), 120)
        for i, atoms in enumerate(self.data):
            shift_energy=0.0
            for z, c in atoms.counts().items():
                # treats all elements as dummy atom type 0
                if self.shift == 'opt-single':
                    atom_types.add(0)
                    atom_counts[i, 0] += c
                else:
                    atom_types.add(z)
                    atom_counts[i, z] = c

                if self.shift == 'pre':
                    shift_energy+=self.pre_energy_shift[z]*c
            _energies=atoms.target_energy.view(-1)-shift_energy
            energies.append(_energies)
            forces.append(atoms.target_forces.view(-1))
        energies = torch.cat(energies)
        forces = torch.cat(forces)
        targets = torch.cat([energies, forces])
        size = targets.numel()
        atom_types = sorted(atom_types)

        # *** kernel matrices ***
        # main block:
        kern = torch.cat([self.Ke, self.Kf])

        # Constant-energy-shift block:
        ntypes = len(atom_types)
        ke_shift = atom_counts[:, atom_types].view(-1,1) if self.shift == 'opt-single' else atom_counts[:, atom_types]
        kf_shift = torch.zeros(self.Kf.size(0), ntypes)
        kern_2 = torch.cat([ke_shift, kf_shift])

        # patch all blocks:
        if self.shift == 'opt' or self.shift == 'opt-single':
            kern = torch.cat([kern, kern_2], dim=1)

        # *** legacy stuff ***
        if not hasattr(self, "_noise"):
            self._noise = {}

        if self.sigma_reg == "default":
            scale = {}
            max_noise=0.99
            if "all" not in self._noise:
                self._noise["all"] = to_inf_inf(self.gp.noise.signal.detach())
            scale["all"] = self.M.diag().mean() * max_noise
            sigma = to_0_1(self._noise["all"]) * scale["all"]
        elif self.sigma_reg == "am":
            if self.multi_mu is None: 
                sigma=0.01
            else:
                sigma = self.smooth_sigma()
        elif isinstance(self.sigma_reg, (int, float)):
            sigma=self.sigma_reg
        else:
            sigma = 0.01

        self.scaled_noise = {"all": sigma}
        #chol = torch.linalg.cholesky(self.M)
        chol,_ = jitcholesky(self.M)

        self.ridge = torch.tensor(0.0)
        self.choli = chol.inverse().contiguous()
        self.Mi = self.choli.t() @ self.choli
        chol_size = chol.size(0)
        self.mean._weight = {}
        self.mean.weights = {}

        M_multi=torch.kron(self.M, self.tasks_kern)
        #chol_multi = torch.linalg.cholesky(M_multi)
        chol_multi,_ = jitcholesky(M_multi)

        choli_multi = chol_multi.inverse().contiguous()
        chol_multi_size = chol_multi.size(0)

        """
        As the optimization of the intertask correlations is time-consuming,
        an alternative is to (acvtively) optimize the intertask correlations over the course of the simulation.
        """
        if self.tasks_kern_optimization is True:
            # decoupled optimizer for mu and W
            # 1. Initial weights \mu optimization ***
            design = torch.kron(kern, self.tasks_kern)

            # *** sgpr extensions ***
            sgpr = True
            if sgpr:
                _kern = sigma * chol_multi.t()
                if self.shift == 'opt' or self.shift == 'opt-single':
                    _kern_2 = torch.zeros(chol_multi_size, self.tasks*ntypes)
                    _kern = torch.cat([_kern, _kern_2], dim=1)
                design = torch.cat([design, _kern])
                _targets = torch.zeros(chol_multi_size)
                targets = torch.cat([targets, _targets])

            solution, predictions = least_squares(design, targets, solver=self.algo)
            self.multi_mu = solution
            self.multi_types = {z: i for i, z in enumerate(atom_types)}

            for i, _ in enumerate(range(self.niter_tasks)):
                # 2. Inter-task kernel W=L@L.T optimization ***
                x1 = self.tasks_kern_L[0][0].item()
                x2 = self.tasks_kern_L[1][0].item()
                x3 = self.tasks_kern_L[1][1].item()

                res = minimize(
                    optimize_task_kern_twobytwo,
                    [x1, x2, x3],
                    args=(kern, self.M, sigma, ntypes, solution, targets, True, self.shift, 1.0),
                )

                self.tasks_kern_L[0][0] = res.x[0]
                self.tasks_kern_L[1][0] = res.x[1]
                self.tasks_kern_L[1][1] = res.x[2]
                self.tasks_kern = self.tasks_kern_L @ self.tasks_kern_L.T

                # 3. Re-optimization of weights \mu based on an updated W ***
                design = torch.kron(kern, self.tasks_kern)

                # *** sgpr extensions ***
                sgpr = True
                if sgpr:
                    M_multi=torch.kron(self.M, self.tasks_kern)
                    #chol_multi = torch.linalg.cholesky(M_multi)
                    chol_multi,_ = jitcholesky(M_multi)

                    choli_multi = chol_multi.inverse().contiguous()

                    _kern = sigma * chol_multi.t()
                    if self.shift == 'opt' or self.shift == 'opt-single':
                        _kern_2 = torch.zeros(chol_multi_size, self.tasks*ntypes)
                        _kern = torch.cat([_kern, _kern_2], dim=1)
                    design = torch.cat([design, _kern])

                solution, predictions = least_squares(design, targets, solver=self.algo)
                self.multi_mu = solution
                self.multi_types = {z: i for i, z in enumerate(atom_types)}

        else:
            # for predetermined tasks corr
            # self.tasks_kern = tasks_correlation(forces.view(self.tasks,-1),corr_coef='pearson')
            
            self.tasks_kern = torch.eye(self.tasks)
            
            # 1. Initial weights \mu optimization ***
            design = torch.kron(kern, self.tasks_kern)

            # *** sgpr extensions ***
            sgpr = True
            if sgpr:
                _kern = sigma * chol_multi.t()
                if self.shift == 'opt' or self.shift == 'opt-single':
                    _kern_2 = torch.zeros(chol_multi_size, self.tasks*ntypes)
                    _kern = torch.cat([_kern, _kern_2], dim=1)
                design = torch.cat([design, _kern])
                _targets = torch.zeros(chol_multi_size)
                targets = torch.cat([targets, _targets])

            solution, predictions = least_squares(design, targets, solver=self.algo)
            self.multi_mu = solution
            self.multi_types = {z: i for i, z in enumerate(atom_types)}

            # np corr
            # import numpy as np
            # self.tasks_kern = np.corrcoef(forces.view(-1, self.tasks)[:,0], forces.view(-1, self.tasks)[:,1])
            # self.tasks_kern = torch.from_numpy(self.tasks_kern)

        # *** stats ***
        # TODO: per-task vscales?
        split = chol_size * self.tasks
        self.mu = solution[:split].reshape(-1, self.tasks)
        self.make_stats((targets[:size], predictions[:size]))

    def predict_multitask_energies(self, kern, local_numbers):
        assert kern.size(0) == len(local_numbers)
        ntypes = len(self.multi_types)
        kern_shift = torch.zeros(kern.size(0), ntypes)

        shift_energy=0.0
        for i, z in enumerate(local_numbers):
            if self.shift == 'opt-single':
                kern_shift[i, 0] = 1.0
            else:    
                if z in self.multi_types:
                    kern_shift[i, self.multi_types[z]] = 1.0
                    if self.shift == 'pre':
                        shift_energy+=self.pre_energy_shift[z]
                else:
                    pass
                    # raise RuntimeError(f'unseen atomic number {z}')

        if self.shift == 'opt' or self.shift == 'opt-single':
            kern = torch.cat([kern, kern_shift], dim=1)
        kern = torch.kron(kern, self.tasks_kern)
        energies = (kern @ self.multi_mu).reshape(-1, self.tasks).sum(dim=0)
        if self.shift == 'pre':
            energies += shift_energy 
        return [e for e in energies]

    def smooth_sigma(self, large_model_size=10, min_sigma = 0.01):
        """
        Adaptive sigma function
        Multitask is susceptible to the choice of sigma. 
        Therefore, this is to prevent the underfitting at the initial phase of training by setting sigma to 0.01.
        And as the training progresses, the sigma saturates to the L2 norm of multi_mu to prevent overfitting.
        """
        model_size = self.ndata
        sigmoid_transition = custom_sigmoid((model_size - large_model_size))
        max_sigma = self.alpha_reg * torch.norm(self.multi_mu, p=2).item()

        sigma = min_sigma + sigmoid_transition * (max_sigma - min_sigma)
        return sigma


def optimize_task_kern_twobytwo(x, kern, M, sigma, ntypes, solution, targets, tasks_reg=False, shift='opt', lmbda=0.1):
    """
    A toy function that measures the error of multitask model
    for a given x in 2 tasks setting

    - tasks_reg: This tag activates the regularization over the tasks kernel matrix. 
    """
    # Prior matrix
    I = torch.eye(2)

    tasks_kern_L_np = np.array([[x[0], 0.0], [x[1], x[2]]], dtype="float64")
    tasks_kern_L = torch.from_numpy(tasks_kern_L_np)
    tasks_kern = tasks_kern_L @ tasks_kern_L.T

    M_multi=torch.kron(M, tasks_kern)
    #chol_multi = torch.linalg.cholesky(M_multi)
    chol_multi, _ = jitcholesky(M_multi)

    design = torch.kron(kern, tasks_kern)

    # *** sgpr extensions ***
    sgpr = True
    if sgpr:
        _kern = sigma * chol_multi.t()
        if shift == 'opt' or shift == 'opt-single':
            _kern_2 = torch.zeros(chol_multi.size(0), tasks_kern.size(0)*ntypes)
            _kern = torch.cat([_kern, _kern_2], dim=1)
        design = torch.cat([design, _kern])

    pred = design @ solution
    err = (pred - targets).abs().mean()

    # Add L2 regularization to encourage tasks_kern_L close to I
    if tasks_reg is True:
        reg_term = lmbda * torch.norm(tasks_kern_L - I)**2
        err += reg_term

    return err.numpy()

def least_squares(design, targets, trials=1, solver="gels", mode="debug"):
    """
    Minimizes
        || design @ solution - targets ||
    using torch.linalg.lstsq.
    Returns
        solution, predictions (= design @ solution)
    Since torch.linalg.lstsq is non-deteministic,
    the best solution of "trials" is return.
    As the design (kern \\otimes tasks_kern) matrix could fall into
    ill-conditioned form, xGELSY and xGELS fail often (for some reason - update the origin).
    - https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
    - https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/lse/lse_drllsp.htm

    For debugging, select debug mode.
        - debug: print the rank of the matrix
        - default: do nothing
    """

    # Compute the rank of the matrix
    rank = torch.linalg.matrix_rank(design)
    
    # Check if the matrix is full-rank
    num_rows, num_cols = design.shape
    is_full_rank = rank == min(num_rows, num_cols)
    
    # print if debug mode
    if mode == "debug":
        if is_full_rank: 
            matrix_ranklog('Matrix is full-rank.\n')
            matrix_ranklog(f'Matrix rank: {rank} min_rows_cols: {min(num_rows, num_cols)}\n')
        else: 
            matrix_ranklog('Matrix is not full-rank.\n')
            matrix_ranklog(f'Matrix rank: {rank} min_rows_cols: {min(num_rows, num_cols)}\n')

    best = None
    predictions = None
    for _ in range(trials):
        sol = torch.linalg.lstsq(design, targets, driver=solver)
        pred = design @ sol.solution
        err = (pred - targets).abs().mean()
        if not best or err < best:
            solution = sol.solution
            predictions = pred
            best = err
    return solution, predictions

def matrix_ranklog(mssge, mode="a"):
    if not distrib.is_initialized() or distrib.get_rank() == 0:
        with open("matrix_rank.log", mode) as f:
            f.write(f"{mssge}\n")

def tasks_correlation(f, corr_coef="pearson"):
    """
    Pre-defined correlation coefficient matrix.
    The optimization of intertask correlation coefficient matrix could be time-consuming.
    We use a bare correlation coefficient matrix based on predicted forces without optimization.
    Or the obtained correlation coefficient matrix is used as the first guess.
    """
    if corr_coef == "pearson":
        W = torch.corrcoef(f)
    else:
        a = f.t() @ f
        b = a.diag().sqrt()[None]
        W = a / (b * b.t())
    return W


def to_0_1(x):
    return 1 / x.neg().exp().add(1.0)

def to_inf_inf(y):
    return (y / y.neg().add(1.0)).log()

def custom_sigmoid(x):
    return 1 / (1 + math.exp(-x))
