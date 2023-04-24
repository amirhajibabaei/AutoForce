# +
import numpy as np
import torch
from scipy.optimize import minimize
import math
import sympy

import theforce.distributed as distrib
from theforce.regression.gppotential import PosteriorPotential


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

    def __init__(
        self, tasks, tasks_kern_optimization, niter_tasks, algo, sigma_reg=None, alpha_reg=0.001, *args, **kwargs
    ):
        """
        tasks: The number of potential energy surface to learn
        tasks_kern_optimization: Flag for tasks_kern optimization 
        niter_tasks: The number of tasks kernel optimization iteration
        algo: torch linear algebra algorithm
        tasks_kern_L: Lower triangle kernel matrix
        sigma_reg: Sigma regularization scheme
            - None:  Constant regularization 
            - am: Adaptive regularization over multi_mu 
            - default: Default single task sigma

        - Energy shift: We shift the average energy average for better training and 
                        for transferrability of the model to extended systems 

            - None: no energy shift
            - opt: constant energy shift term per element is determined from a SGPR equation.          
            - pre: Use a pre-calculated isolated atom energy of each element type
            - preopt: Based on an atom energy, optimize the shift level 
        """

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
        self.shift = 'opt'
        self.pre_energy_shift={0: 0.0, 1: -13.6766048214, 8: -429.072746017}

    def make_munu(self, *args, **kwargs):

        # *** targets ***
        energies, forces = [], []
        atom_types = set()
        # for dummy atom type Z
        atom_counts = torch.zeros(len(self.data), 120)
        for i, atoms in enumerate(self.data):
            shift_energy=0.0
            for z, c in atoms.counts().items():
                atom_types.add(z)
                atom_counts[i, z] = c
                if self.shift is 'pre':
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
        ke_shift = atom_counts[:, atom_types]
        kf_shift = torch.zeros(self.Kf.size(0), ntypes)
        kern_2 = torch.cat([ke_shift, kf_shift])

        # patch all blocks:
        if self.shift is 'opt' or 'preopt':
            kern = torch.cat([kern, kern_2], dim=1)

        # *** legacy stuff ***
        if not hasattr(self, "_noise"):
            self._noise = {}

        if self.sigma_reg is "default":
            scale = {}
            max_noise=0.99
            if "all" not in self._noise:
                self._noise["all"] = to_inf_inf(self.gp.noise.signal.detach())
            scale["all"] = self.M.diag().mean() * max_noise
            sigma = to_0_1(self._noise["all"]) * scale["all"]
        elif self.sigma_reg is "am":
            if self.multi_mu is None: 
                sigma=0.01
            else:
                sigma = self.smooth_sigma()
        elif isinstance(self.sigma_reg, (int, float)):
            sigma=self.sigma_reg
        else:
            sigma = 0.01

        self.scaled_noise = {"all": sigma}
        chol = torch.linalg.cholesky(self.M)
        self.ridge = torch.tensor(0.0)
        self.choli = chol.inverse().contiguous()
        self.Mi = self.choli.t() @ self.choli
        chol_size = chol.size(0)
        self.mean._weight = {}
        self.mean.weights = {}

        M_multi=torch.kron(self.M, self.tasks_kern)
        chol_multi = torch.linalg.cholesky(M_multi)
        choli_multi = chol_multi.inverse().contiguous()
        chol_multi_size = chol_multi.size(0)

        # *** tasks kernel ***
        # covariance between tasks:
        # in principle this should be optimized as a hyper parameter.
        # for now we set it equal to correlation of forces in different
        # tasks.

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
                if self.shift is 'opt' or 'preopt':
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
                    args=(kern, self.M, sigma, ntypes, solution, targets),
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
                    chol_multi = torch.linalg.cholesky(M_multi)
                    choli_multi = chol_multi.inverse().contiguous()

                    _kern = sigma * chol_multi.t()
                    if self.shift is 'opt':
                        _kern_2 = torch.zeros(chol_multi_size, self.tasks*ntypes)
                        _kern = torch.cat([_kern, _kern_2], dim=1)
                    design = torch.cat([design, _kern])

                solution, predictions = least_squares(design, targets, solver=self.algo)
                self.multi_mu = solution
                self.multi_types = {z: i for i, z in enumerate(atom_types)}

                #design = torch.kron(kern, self.tasks_kern)
                #solution, predictions = least_squares(design, targets, solver=self.algo)
                #self.multi_mu = solution
                #self.multi_types = {z: i for i, z in enumerate(atom_types)}
                # print(f'{i} Optimized tasks corrs: {self.tasks_kern}, Lower: {self.tasks_kern_L}, error: {res.fun}')
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
                if self.shift is 'opt':
                    _kern_2 = torch.zeros(chol_multi_size, self.tasks*ntypes)
                    _kern = torch.cat([_kern, _kern_2], dim=1)
                design = torch.cat([design, _kern])
                _targets = torch.zeros(chol_multi_size)
                targets = torch.cat([targets, _targets])

            solution, predictions = least_squares(design, targets, solver=self.algo)
            self.multi_mu = solution
            self.multi_types = {z: i for i, z in enumerate(atom_types)}

            # for independent tasks, set:
            # self.tasks_kern = torch.eye(self.tasks)
            # a small ridge maybe needed in case tasks are identical:
            # self.tasks_kern += torch.eye(self.tasks) * 1e-1

            # np corr
            # import numpy as np
            # self.tasks_kern = np.corrcoef(forces.view(-1, self.tasks)[:,0], forces.view(-1, self.tasks)[:,1])
            # self.tasks_kern = torch.from_numpy(self.tasks_kern)

            # *** solution ***
            #design = torch.kron(kern, self.tasks_kern)
            #solution, predictions = least_squares(design, targets, solver=self.algo)
            #self.multi_mu = solution
            #self.multi_types = {z: i for i, z in enumerate(atom_types)}

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
            if z in self.multi_types:
                kern_shift[i, self.multi_types[z]] = 1.0
                if self.shift is 'pre':
                    shift_energy+=self.pre_energy_shift[z]
            else:
                pass
                # raise RuntimeError(f'unseen atomic number {z}')

        if self.shift is 'opt':
            kern = torch.cat([kern, kern_shift], dim=1)
        kern = torch.kron(kern, self.tasks_kern)
        energies = (kern @ self.multi_mu).reshape(-1, self.tasks).sum(dim=0)
        if self.shift is 'pre':
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


def optimize_task_kern_twobytwo(x, kern, M, sigma, ntypes, solution, targets, tasks_reg=False, shift='opt'):
    """
    A toy function that measures the error of multitask model
    for a given x in 2 tasks setting

    - tasks_reg: This tag activates the regularization over the tasks kernel matrix. 
    """
    tasks_kern_L_np = np.array([[x[0], 0.0], [x[1], x[2]]], dtype="float64")
    tasks_kern_L = torch.from_numpy(tasks_kern_L_np)
    tasks_kern = tasks_kern_L @ tasks_kern_L.T

    # Add L2 regularization to diagonal elements of tasks_kern
    if tasks_reg is True:
        alpha=0.01
        diag = torch.diagonal(tasks_kern)
        diag_reg = alpha*(diag-1)**2
        tasks_kern += torch.diag(diag_reg)

    M_multi=torch.kron(M, tasks_kern)
    chol_multi = torch.linalg.cholesky(M_multi)

    design = torch.kron(kern, tasks_kern)

    # *** sgpr extensions ***
    sgpr = True
    if sgpr:
        _kern = sigma * chol_multi.t()
        if shift is 'opt':
            _kern_2 = torch.zeros(chol_multi.size(0), tasks_kern.size(0)*ntypes)
            _kern = torch.cat([_kern, _kern_2], dim=1)
        design = torch.cat([design, _kern])

    pred = design @ solution
    err = (pred - targets).abs().mean()

    # Add L2 regularization term to diagonal elements of tasks_kern
    if tasks_reg is True:
        tasks_kern_diag = torch.diagonal(tasks_kern)
        tasks_kern_diag_reg = alpha * tasks_kern_diag ** 2
        err += tasks_kern_diag_reg.sum()

    return err.numpy()

def least_squares(design, targets, trials=1, solver="gels"):
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
    """

    # Compute the rank of the matrix
    rank = torch.matrix_rank(design)
    
    # Check if the matrix is full-rank
    num_rows, num_cols = design.shape
    is_full_rank = rank == min(num_rows, num_cols)
    
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


def least_squares_gram(design, targets, trials=1, solver="gels"):
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
    """

#    if rref is True: 
#        rref_matrix, pivot_columns = sympy.Matrix(design.numpy()).rref()
#        matrix_ranklog(f'pivot columns: {pivot_columns}\n')
#        design = design[:, pivot_columns]

    # Compute the Gram matrix (A^T * A)
    design2 = design.t() @ design
    
    # Regularize the Gram matrix by adding a multiple of the identity matrix
    alpha = 1e-6
    design2 = design2 + alpha * torch.eye(design2.shape[0])
    
    # Solve the regularized linear system
    targets2 = design.t() @ targets

    # Compute the rank of the matrix
    rank = torch.matrix_rank(design2)
    
    # Check if the matrix is full-rank
    num_rows, num_cols = design2.shape
    is_full_rank = rank == min(num_rows, num_cols)
    
    if is_full_rank: 
        matrix_ranklog('Matrix is full-rank.\n')
        matrix_ranklog(f'Matrix rank: {rank} min_rows_cols: {min(num_rows, num_cols)}\n')
    else: 
        matrix_ranklog('Matrix is not full-rank.\n')
        matrix_ranklog(f'Matrix rank: {rank} min_rows: {num_rows} min_cols: {num_cols}\n')

    best = None
    predictions = None
    for _ in range(trials):
        sol = torch.linalg.lstsq(design2, targets2, driver=solver)
        pred = design @ sol.solution
        err = (pred - targets).abs().mean()
        if not best or err < best:
            solution = sol.solution
            predictions = pred
            best = err
    return solution, predictions

def matrix_ranklog(mssge, mode="a"):
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
