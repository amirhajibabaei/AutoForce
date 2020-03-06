import torch
from torch.nn import Module
from torch.distributions import MultivariateNormal, LowRankMultivariateNormal
from theforce.regression.kernel import White
from theforce.regression.algebra import jitcholesky, projected_process_auxiliary_matrices_D
from theforce.util.util import iterable, mkdir_p, safe_dirname
from theforce.optimize.optimizers import ClampedSGD
import copy
import os
import functools


class EnergyForceKernel(Module):

    def __init__(self, similaritykernels):
        super().__init__()
        self.kernels = iterable(similaritykernels)
        self.params = [par for kern in self.kernels for par in kern.params]

    def forward(self, first, second=None, cov='energy_energy', inducing=None):
        sec = first if second is None else second
        if inducing is None:
            return getattr(self, cov)(first, sec)
        else:
            middle = getattr(self, 'energy_energy')(inducing, inducing)
            chol, _ = jitcholesky(middle)
            invchol = chol.inverse()
            lcov, rcov = cov.split('_')
            left = getattr(self, lcov+'_energy')(first, inducing) @ invchol.t()
            if second is None and rcov == lcov:
                right = left.t()
            else:
                right = invchol @ getattr(self, 'energy_'+rcov)(inducing, sec)
            return left, right

    def energy_energy(self, first, second):
        return self.base_kerns(first, second, 'func')

    def forces_energy(self, first, second):
        return -self.base_kerns(first, second, 'leftgrad')

    def energy_forces(self, first, second):
        return -self.base_kerns(first, second, 'rightgrad')

    def forces_forces(self, first, second):
        return self.base_kerns(first, second, 'gradgrad')

    def base_kerns(self, first, second, operation):
        return torch.stack([kern(first, second, operation=operation)
                            for kern in self.kernels]).sum(dim=0)

    # diagonal elements:
    def diag(self, data, operation='energy'):
        return getattr(self, operation+'_diag')(data)

    def full_diag(self, data):
        return self.energy_forces_diag(data)

    def energy_forces_diag(self, data):
        return torch.cat([self.energy_diag(data), self.forces_diag(data)])

    def energy_diag(self, data):
        return self.base_kerns_diag(data, 'func')

    def forces_diag(self, data):
        return self.base_kerns_diag(data, 'gradgrad')

    def base_kerns_diag(self, data, operation):
        return torch.stack([kern.diag(data, operation=operation)
                            for kern in self.kernels]).sum(dim=0)

    @property
    def method_caching(self):
        return [kern.method_caching if hasattr(kern, 'method_caching') else False
                for kern in self.kernels]

    @method_caching.setter
    def method_caching(self, value):
        if hasattr(value, '__iter__'):
            val = value
        else:
            val = len(self.kernels)*[value]
        for kern, v in zip(*[self.kernels, val]):
            kern.method_caching = v

    def clear_cached(self):
        for kern in self.kernels:
            try:
                kern.cached.clear()
            except AttributeError:
                pass

    @property
    def state_args(self):
        return '[{}]'.format(', '.join([kern.state for kern in self.kernels]))

    @property
    def state(self):
        return 'EnergyForceKernel({})'.format(self.state_args)


class GaussianProcessPotential(Module):

    def __init__(self, kernels, noise=White(signal=0.01, requires_grad=True), parametric=None):
        super().__init__()
        self.kern = EnergyForceKernel(kernels)
        self.noise = noise
        for i, kern in enumerate(self.kern.kernels):
            kern.name = 'kern_{}'.format(i)
        self.parametric = parametric

    @property
    def params(self):
        p = self.kern.params + self.noise.params
        if self.parametric is not None:
            p += self.parametric.unique_params
        return p

    @property
    def species(self):
        return [kern.a for kern in self.kern.kernels]

    @property
    def requires_grad(self):
        return [p.requires_grad for p in self.params]

    @requires_grad.setter
    def requires_grad(self, value):
        for p in self.params:
            p.requires_grad = value

    def forward(self, data, inducing=None):
        if inducing is None:
            L = torch.cat([self.kern(data, cov='energy_energy'),
                           self.kern(data, cov='forces_energy')], dim=0)
            R = torch.cat([self.kern(data, cov='energy_forces'),
                           self.kern(data, cov='forces_forces')], dim=0)
            return MultivariateNormal(torch.zeros(L.size(0)),
                                      covariance_matrix=torch.cat([L, R], dim=1) +
                                      torch.eye(L.size(0))*self.noise.signal**2)
        else:
            Q = torch.cat([self.kern(data, cov='energy_energy', inducing=inducing)[0],
                           self.kern(data, cov='forces_forces', inducing=inducing)[0]], dim=0)
            return LowRankMultivariateNormal(torch.zeros(Q.size(0)), Q, self.diagonal_ridge(data))

    def diagonal_ridge(self, data, operation='full'):
        s = self.noise.signal**2
        e_diag = torch.tensor(data.natoms, dtype=s.dtype) * s
        f_diag = torch.ones(3*sum(data.natoms)) * s
        if operation == 'energy':
            return e_diag
        elif operation == 'forces':
            return f_diag
        elif operation == 'full':
            return torch.cat([e_diag, f_diag])

    def mean(self, data, forces=True, cat=True):
        if self.parametric is None:
            if forces:
                if cat:
                    return 0
                else:
                    return 0, 0
            else:
                return 0
        else:
            e = [self.parametric(sys, forces=forces) for sys in iterable(data)]
            if forces:
                e, f = zip(*e)
                e = torch.cat([_e.view(-1) for _e in e])
                f = torch.cat(f).view(-1)
                if cat:
                    return torch.cat([e, f])
                else:
                    return e, f
            else:
                return torch.cat([_e.view(-1) for _e in e])

    def Y(self, data):
        y = torch.cat([torch.tensor([sys.target_energy for sys in data])] +
                      [sys.target_forces.view(-1) for sys in data])
        return y - self.mean(data)

    def loss(self, data, Y=None, inducing=None, logprob_loss=True, cov_loss=False):
        p = self(data, inducing=inducing)
        if hasattr(p, 'cov_factor'):
            if cov_loss:
                covariance_loss = 0.5 * ((self.kern.diag(data, 'full') - p.cov_factor.pow(2).sum(dim=-1)
                                          )/self.diagonal_ridge(data)).sum()
            else:
                covariance_loss = 0
        else:
            covariance_loss = 0
        if logprob_loss:
            lp_loss = -p.log_prob(self.Y(data) if Y is None else Y)
        else:
            lp_loss = 0
        return lp_loss + covariance_loss

    @property
    def method_caching(self):
        return self.kern.method_caching

    @method_caching.setter
    def method_caching(self, value):
        self.kern.method_caching = value

    def clear_cached(self, X=None):
        if X is None:
            self.kern.clear_cached()
        else:
            for x in iterable(X):
                if hasattr(x, 'UID'):
                    UID = x.UID()
                    for a in self.cached:
                        for b in a.values():
                            for c in list(b.keys()):
                                if UID in c:
                                    del b[c]

    @property
    def cached(self):
        return [kern.cached if hasattr(kern, 'cached') else {}
                for kern in self.kern.kernels]

    @cached.setter
    def cached(self, values):
        for kern, val in zip(*[self.kern.kernels, value]):
            kern.cached = val

    def del_cached(self):
        for kern in self.kern.kernels:
            if hasattr(kern, 'cached'):
                del kern.cached

    def attach_process_group(self, group=torch.distributed.group.WORLD):
        for kern in self.kern.kernels:
            kern.process_group = group

    def detach_process_group(self):
        for kern in self.kern.kernels:
            del kern.process_group

    @property
    def state_args(self):
        return '{}, noise={}, parametric={}'.format(self.kern.state_args, self.noise.state,
                                                    self.parametric)

    @property
    def state(self):
        return 'GaussianProcessPotential({})'.format(self.state_args)

    def __repr__(self):
        return self.state

    def to_file(self, file, flag='', mode='a'):
        from theforce.util.util import one_liner
        with open(file, mode) as f:
            f.write('\n\n\n#flag: {}\n'.format(flag))
            f.write(one_liner(self.state))


def context_setting(method):

    @functools.wraps(method)
    def wrapper(self, *args, use_caching=False, enable_grad=False, **kwargs):
        caching_status = self.gp.method_caching
        self.gp.method_caching = use_caching
        with torch.set_grad_enabled(enable_grad):
            result = method(self, *args, **kwargs)
        self.gp.method_caching = caching_status
        return result

    return wrapper


class PosteriorPotential(Module):

    def __init__(self, gp, data, inducing=None, group=None, **setting):
        super().__init__()
        self.gp = gp
        if group is not None:
            self.attach_process_group(group)
        else:
            self.is_distributed = False
        self.set_data(data, inducing=inducing, **setting)

    @context_setting
    def set_data(self, data, inducing=None):
        self.data = data
        if inducing is None:
            raise RuntimeWarning(
                'This (inducing=None) has not been used in long while!')
            p = self.gp(data, inducing)
            self.X = copy.deepcopy(data)  # TODO: consider not copying
            self.mu = p.precision_matrix @ (gp.Y(data)-p.loc)
            self.nu = p.precision_matrix
            self.has_target_forces = True
        else:
            X = inducing.subset(self.gp.species)
            self.Ke = self.gp.kern(data, X, cov='energy_energy')
            self.Kf = self.gp.kern(data, X, cov='forces_energy')
            self.M = self.gp.kern(X, X, cov='energy_energy')
            self.X = X
            self.make_munu()
            self.has_target_forces = False

    @property
    def inducing(self):
        return self.X

    @property
    def K(self):
        return torch.cat([self.Ke, self.Kf], dim=0)

    def make_munu(self):
        self.mu, self.nu, self.ridge, self.choli = projected_process_auxiliary_matrices_D(
            self.K, self.M, self.gp.Y(self.data), self.gp.diagonal_ridge(self.data), chol_inverse=True)

    @property
    def ref_M(self):
        return self.M + self.ridge*torch.eye(self.M.size(0))

    @context_setting
    def leakage(self, loc):
        a = self.gp.kern(self.X, loc, cov='energy_energy')
        b = self.choli @ a
        c = b.t()@b
        d = self.gp.kern(loc, loc, cov='energy_energy') + self.ridge
        return (1-c/d).view(1)

    def leakages(self, X):
        return torch.cat([self.leakage(x) for x in iterable(X)])

    @context_setting
    def remake_all(self):
        self.set_data(self.data, self.X)

    @context_setting
    def add_data(self, data, remake=True):
        Ke = self.gp.kern(data, self.X, cov='energy_energy')
        Kf = self.gp.kern(data, self.X, cov='forces_energy')
        self.Ke = torch.cat([self.Ke, Ke], dim=0)
        self.Kf = torch.cat([self.Kf, Kf], dim=0)
        self.data += data
        if remake:
            self.make_munu()

    @context_setting
    def add_inducing(self, X, remake=True):
        assert X.number in self.gp.species
        Ke = self.gp.kern(self.data, X, cov='energy_energy')
        Kf = self.gp.kern(self.data, X, cov='forces_energy')
        self.Ke = torch.cat([self.Ke, Ke], dim=1)
        self.Kf = torch.cat([self.Kf, Kf], dim=1)
        a = self.gp.kern(self.X, X, cov='energy_energy')
        b = self.gp.kern(X, X, cov='energy_energy')
        self.M = torch.cat(
            [torch.cat([self.M, a.t()]), torch.cat([a, b])], dim=1)
        self.X += X
        if remake:
            self.make_munu()

    def pop_1data(self, remake=True, clear_cached=True):
        self.Ke = self.Ke[:-1]
        self.Kf = self.Kf[:-3*self.data[-1].natoms]
        if clear_cached:
            self.gp.clear_cached([self.data.X[-1]])
        del self.data.X[-1]
        if remake:
            self.make_munu()

    def pop_1inducing(self, remake=True, clear_cached=True):
        self.Ke = self.Ke[:, :-1]
        self.Kf = self.Kf[:, :-1]
        self.M = self.M[:-1, :-1]
        if clear_cached:
            self.gp.clear_cached([self.X.X[-1]])
        del self.X.X[-1]
        if remake:
            self.make_munu()

    def downsize(self, n, m):
        while len(self.data) > n:
            self.pop_1data()
        while len(self.X) > m:
            self.pop_1inducing()

    def add_1atoms(self, atoms, ediff, fdiff):
        kwargs = {'use_caching': True}
        e1 = self([atoms], **kwargs)
        if fdiff < float('inf'):
            f1 = self([atoms], 'forces', **kwargs)
        self.add_data([atoms], **kwargs)
        e2 = self([atoms], **kwargs)
        de = abs(e1-e2)
        if fdiff < float('inf'):
            f2 = self([atoms], 'forces', **kwargs)
            df = (f2-f1).abs().max()
        else:
            df = 0
        blind = torch.cat([e1, e2]).allclose(torch.zeros(1))
        if de < ediff and df < fdiff and not blind:
            self.pop_1data(clear_cached=True)
        return de, df

    def add_1inducing(self, _loc, ediff, detach=True):
        if _loc.number not in self.gp.species:
            return 0
        kwargs = {'use_caching': True}
        if detach:
            loc = _loc.detach()
            loc.stage(self.gp.kern.kernels, dont_save_grads=True)
        else:
            loc = _loc
        e1 = self(loc, **kwargs)
        self.add_inducing(loc, **kwargs)
        e2 = self(loc, **kwargs)
        de = abs(e1-e2)
        blind = torch.cat([e1, e2]).allclose(torch.zeros(1))
        if de < ediff and not blind:
            self.pop_1inducing(clear_cached=True)
        return de

    def add_ninducing(self, _locs, ediff, detach=True, descending=True):
        locs = [loc for loc in _locs if loc.number in self.gp.species]
        if descending:
            leaks = self.leakages(locs)
            q = torch.argsort(leaks, descending=True)
        else:
            q = torch.arange(len(locs))
        added = 0
        for k in q:
            loc = locs[k]
            _ediff = ediff if len(self.X) > 1 else torch.finfo().tiny
            change = self.add_1inducing(loc, _ediff, detach=detach)
            if change >= ediff:
                added += 1
            else:
                if descending:
                    break
        return added, change

    def select_inducing(self, indices, deleted=None, remake=True):
        i = torch.as_tensor(indices)
        self.Ke = self.Ke.index_select(1, i)
        self.Kf = self.Kf.index_select(1, i)
        self.M = self.M.index_select(0, i).index_select(1, i)
        if deleted:
            self.gp.clear_cached([self.X.X[j] for j in deleted])
        self.X.X = [self.X.X[j] for j in i]
        if remake:
            self.make_munu()

    def attach_process_group(self, *args, **kwargs):
        self.gp.attach_process_group(*args, **kwargs)
        self.is_distributed = True

    def detach_process_group(self, *args, **kwargs):
        self.gp.detach_process_group(*args, **kwargs)
        self.is_distributed = False

    def train(self, *args, **kwargs):
        train_gpp(self.gp, *args, **kwargs)

    def save(self, file, supress_warnings=True):
        import warnings
        cached = self.gp.cached
        self.gp.del_cached()
        data = self.data
        del self.data
        with warnings.catch_warnings():
            if supress_warnings:
                warnings.simplefilter("ignore")
            torch.save(self, file)
        self.data = data
        self.gp.cahced = cached

    def to_folder(self, folder, info=None, overwrite=True, supress_warnings=True):
        if not overwrite:
            folder = safe_dirname(folder)
        mkdir_p(folder)
        with open(os.path.join(folder, 'cutoff'), 'w') as file:
            try:
                file.write('{}\n'.format(self._cutoff))
            except:
                file.write('{}\n'.format(self.data[0].cutoff))
        self.data.to_traj(os.path.join(folder, 'data.traj'))
        self.X.to_traj(os.path.join(folder, 'inducing.traj'))
        self.gp.to_file(os.path.join(folder, 'gp'))
        self.save(os.path.join(folder, 'model'),
                  supress_warnings=supress_warnings)
        # info
        with open(os.path.join(folder, 'info'), 'w') as file:
            file.write('data: {}, inducing: {}\n'.format(
                len(self.data), len(self.X)))
            if info is not None:
                if type(info) == str:
                    file.write('{}\n'.format(info))
                elif hasattr(info, '__iter__'):
                    for inf in info:
                        file.write('{}\n'.format(inf))
                else:
                    file.write('{}\n'.format(info))

    @context_setting
    def forward(self, test, quant='energy', variance=False, all_reduce=False):
        shape = {'energy': (-1,), 'forces': (-1, 3)}
        A = self.gp.kern(test, self.X, cov=quant+'_energy')
        if self.has_target_forces:
            A = torch.cat([A, self.gp.kern(test, self.X, cov=quant+'_forces')],
                          dim=1)
        if quant == 'energy':
            mean = self.gp.mean(test, forces=False)
        else:
            _, mean = self.gp.mean(test, forces=True, cat=False)
        out = (mean + A @ self.mu).view(*shape[quant])
        if all_reduce:
            torch.distributed.all_reduce(out)
        if variance:
            if all_reduce:
                raise NotImplementedError(
                    'all_reduce with variance=True is not implemented')
            var = (self.gp.kern.diag(test, quant) -
                   (A @ self.nu @ A.t()).diag()).view(*shape[quant])
            return out, var
        else:
            return out

    @context_setting
    def predict(self, test, variance=False):
        if self.gp.parametric is not None:
            raise NotImplementedError(
                'this method is not updated to include parametric potential')
        A = self.gp.kern(test, self.X, cov='energy_energy')
        B = self.gp.kern(test, self.X, cov='forces_energy')
        if self.has_target_forces:
            A = torch.cat([A, self.gp.kern(test, self.X, cov='energy_forces')],
                          dim=1)
            B = torch.cat([B, self.gp.kern(test, self.X, cov='forces_forces')],
                          dim=1)

        energy = A @ self.mu
        forces = B @ self.mu

        out = (energy, forces.view(-1, 3))

        if variance:
            energy_var = (self.gp.kern.diag(test, 'energy') -
                          (A @ self.nu @ A.t()).diag())
            forces_var = (self.gp.kern.diag(test, 'forces') -
                          (B @ self.nu @ B.t()).diag())
            out += (energy_var, forces_var.view(-1, 3))

        return out


def PosteriorPotentialFromFolder(folder, load_data=True, update_data=True):
    from theforce.descriptor.atoms import AtomsData
    from theforce.util.caching import strip_uid
    self = torch.load(os.path.join(folder, 'model'))
    strip_uid(self.X)
    with open(os.path.join(folder, 'cutoff'), 'r') as file:
        cutoff = float(file.readline().split()[0])
        self._cutoff = cutoff
    if load_data:
        self.data = AtomsData(traj=os.path.join(folder, 'data.traj'))
        if update_data:
            self.data.update(cutoff=cutoff, descriptors=self.gp.kern.kernels)
    return self


def train_gpp(gp, X, inducing=None, steps=10, lr=0.1, Y=None, logprob_loss=True, cov_loss=False,
              move=0.1, shake=0):
    if not logprob_loss and not cov_loss:
        raise RuntimeError('both loss terms are ignored!')

    if not hasattr(gp, 'optimizer'):
        gp.optimizer = torch.optim.Adam([{'params': gp.params}], lr=lr)
        if inducing is not None and inducing.trainable and not hasattr(inducing, 'optimizer'):
            inducing.optimizer = ClampedSGD(
                [{'params': inducing.params}], lr=move)

    caching_status = gp.method_caching
    gp.method_caching = False
    gp.clear_cached()

    for _ in range(steps):
        if inducing is not None and inducing.trainable:
            if shake > 0:
                inducing.shake(update=False)
            inducing.optimizer.zero_grad()
            inducing.update_nl_if_requires_grad(descriptors=gp.kern.kernels)
        gp.optimizer.zero_grad()
        loss = gp.loss(X, Y, inducing, logprob_loss, cov_loss)
        loss.backward()
        gp.optimizer.step()
        if inducing is not None and inducing.trainable:
            inducing.optimizer.step()

    gp.method_caching = caching_status

    report = []
    if inducing is None:
        report += ['Full']
    else:
        report += ['Sparse']
        if cov_loss:
            report += ['Variational-ELBO']
            if not logprob_loss:
                report += ['~only trace of Gram matrix considered']
        else:
            report += ['Projected-Process']
    report = ' '.join(report)
    print('trained for {} steps ({})'.format(steps, report))

