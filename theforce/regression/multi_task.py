# +
from theforce.regression.gppotential import PosteriorPotential
import theforce.distributed as distrib
import torch
from scipy.optimize import minimize
import numpy as np


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

    def __init__(self, tasks, tasks_kern_optimization, niter_tasks, algo, *args, **kwargs):
        """
        tasks: number of potential energy types.
        
        """
        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.tasks_kern_L = torch.eye(self.tasks)
        self.tasks_kern_L += 1e-2
        self.tasks_kern   = torch.eye(self.tasks)
        self.tasks_kern_optimization=tasks_kern_optimization
        self.niter_tasks=niter_tasks
        self.algo=algo

    def make_munu(self, *args, **kwargs):

        # *** targets ***
        energies, forces = [], []
        atom_types = set()
        #for dummy atom type Z
        atom_counts = torch.zeros(len(self.data), 120)
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

        '''
        As the optimization of the intertask correlations is time-consuming, 
        an alternative is to (acvtively) optimize the intertask correlations over the course of the simulation.
        '''
        if self.tasks_kern_optimization is True:
            #self.tasks_kern_L = torch.eye(self.tasks)
            #self.tasks_kern_L += 1e-2

            # decoupled optimizer for mu and W
            ## 1. Initial weights \mu optimization ***
            design = torch.kron(kern, self.tasks_kern)
            solution, predictions = least_squares(design, targets, solver=self.algo)
            self.multi_mu = solution
            self.multi_types = {z: i for i, z in enumerate(atom_types)}

            for i,_ in enumerate(range(self.niter_tasks)):
                # 2. Inter-task kernel W=L@L.T optimization ***
                x1=self.tasks_kern_L[0][0].item()
                x2=self.tasks_kern_L[1][0].item()
                x3=self.tasks_kern_L[1][1].item()
                res=minimize(optimize_task_kern_twobytwo,[x1,x2,x3],args=(kern,solution,targets))
                self.tasks_kern_L[0][0]=res.x[0]
                self.tasks_kern_L[1][0]=res.x[1]
                self.tasks_kern_L[1][1]=res.x[2]
                self.tasks_kern = self.tasks_kern_L@self.tasks_kern_L.T

                # 3. Re-optimization of weights \mu based on an updated W ***
                design = torch.kron(kern, self.tasks_kern)
                solution, predictions = least_squares(design, targets, solver=self.algo)
                self.multi_mu = solution
                self.multi_types = {z: i for i, z in enumerate(atom_types)}
                #print(f'{i} Optimized tasks corrs: {self.tasks_kern}, Lower: {self.tasks_kern_L}, error: {res.fun}')
        else:
            # for predetermined tasks corr
            #self.tasks_kern = tasks_correlation(forces.view(self.tasks,-1),corr_coef='pearson')
            self.tasks_kern = torch.eye(self.tasks)
    
            # for independent tasks, set:
            #self.tasks_kern = torch.eye(self.tasks)
            # a small ridge maybe needed in case tasks are identical:
            #self.tasks_kern += torch.eye(self.tasks) * 1e-1
    
            #np corr
            #import numpy as np
            #self.tasks_kern = np.corrcoef(forces.view(-1, self.tasks)[:,0], forces.view(-1, self.tasks)[:,1])
            #self.tasks_kern = torch.from_numpy(self.tasks_kern) 
    
            # *** solution ***
            design = torch.kron(kern, self.tasks_kern)
            solution, predictions = least_squares(design, targets, solver=self.algo)
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

def optimize_task_kern_twobytwo(x,kern,solution,targets):
    """
    A toy function that measures the error of multitask model
    for a given x in 2 tasks setting
    """
    tasks_kern_L_np=np.array([[x[0],0.],[x[1],x[2]]],dtype='float64')
    tasks_kern_L=torch.from_numpy(tasks_kern_L_np)
    tasks_kern=tasks_kern_L@tasks_kern_L.T
    design = torch.kron(kern, tasks_kern)
    pred = design @ solution
    err = (pred - targets).abs().mean()
    return err.numpy()
    
def least_squares(design, targets, trials=10, solver='gelsd'):
    """
    Minimizes 
        || design @ solution - targets ||
    using torch.linalg.lstsq.
    
    Returns
        solution, predictions (= design @ solution)
        
    Since torch.linalg.lstsq is non-deteministic,
    the best solution of "trials" is return. 

    As the design (kern \otimes tasks_kern) matrix could fall into 
    ill-conditioned form, xGELSY and xGELS fail often (for some reason - update the origin). 

    - https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html#torch.linalg.lstsq
    - https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/lse/lse_drllsp.htm
    """
    best = None
    predictions = None
    for _ in range(trials):
        sol = torch.linalg.lstsq(design,targets,driver=solver)
        pred = design @ sol.solution
        err = (pred - targets).abs().mean()
        if not best or err < best:
            solution = sol.solution
            predictions = pred
            best = err
    return solution, predictions

def tasks_correlation(f,corr_coef='pearson'):
    '''
    Pre-defined correlation coefficient matrix. 
    The optimization of intertask correlation coefficient matrix could be time-consuming.
    We use a bare correlation coefficient matrix based on predicted forces without optimization.
    Or the obtained correlation coefficient matrix is used as the first guess. 
    '''
    if (corr_coef=='pearson'):
        W=torch.corrcoef(f)
    else:
        a = f.t() @ f
        b = a.diag().sqrt()[None]
        W=a / (b * b.t())
    return W
