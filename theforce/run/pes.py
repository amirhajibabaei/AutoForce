
# coding: utf-8

# In[ ]:


import itertools
import os
import random
import numpy as np
import torch


def read_params(**kwargs):

    # default params
    params = {
        # gp
        'numbers': None,
        'cutoff': None,
        'atomic_unit': None,
        'lmax': 2,
        'nmax': 2,
        'exponent': 4,
        'pairkernel': True,
        'soapkernel': True,
        'error': 0.05
        # data
        'path_data': None,
        'path_data_chp': None,
        'path_inducing_chp': None,
        'path_gp_chp': 'gp.chp',
        'ndata': -1,
        'nlocals': -1,
        # other
        'path_log': 'log.txt',
        'test': False
    }

    # read param.txt if it exists
    if os.path.isfile('param.txt'):
        with open('params.txt') as file:
            for line in file.readlines():
                if line.strip() == '':
                    continue
                a, b = (_s.strip() for _s in line.strip().split(':'))
                if a not in params:
                    raise RuntimeWarning(
                        'params.txt: keyword {} is not recognized'.format(a))
                    continue
                if a.startswith('path'):
                    params[a] = b
                else:
                    params[a] = eval('{}'.format(b))

    # read kwargs
    for a, b in kwargs.items():
        if a not in params:
            raise RuntimeWarning(
                'kwargs: keyword {} is not recognized'.format(a))
            continue
        params[a] = b

    return params


def potential_energy_surface(data=None, inducing=None, train=0, append_log=True, **kwargs):
    from theforce.descriptor.atoms import AtomsData, LocalsData, sample_atoms
    from theforce.regression.gppotential import GaussianProcessPotential
    from theforce.regression.gppotential import PosteriorPotential
    from theforce.regression.gppotential import train_gpp
    from theforce.similarity.pair import PairKernel
    from theforce.similarity.soap import SoapKernel
    from theforce.regression.stationary import RBF
    from theforce.math.cutoff import PolyCut
    from theforce.util.util import iterable
    from theforce.regression.kernel import White, Positive, DotProd, Normed, Mul, Pow, Add
    from torch import tensor

    # read params
    params = read_params(**kwargs)
    log = open(params['path_log'], 'a' if append_log else 'w')
    log.write('{}\n'.format(37*'*'))

    # data
    if data is None:
        data = sample_atoms(params['path_data'], size=params['ndata'],
                            chp=params['path_data_chp'])
        data.update(cutoff=params['cutoff'])
        log.write('cutoff: {}\npath_data: {}\nndata: {}\npath_data_chp: {}\n'.format(
            params['cutoff'], params['path_data'], params['ndata'], params['path_data_chp']))
    else:
        # if data is given as kwarg, it already should be cutoff-ed.
        assert len(data[-1]) == data[-1].natoms
        log.write('ndata: {} (kwarg)\n'.format(len(data)))

    # inducing
    if inducing is None:
        if params['nlocals'] == -1:
            inducing = data.to_locals()
            log.write('nlocals: {} (=-1)\n'.format(len(inducing)))
        else:
            inducing = data.sample_locals(params['nlocals'])
            log.write('nlocals: {}\n'.format(params['nlocals']))
    else:
        log.write('nlocals: {} (kwarg)\n'.format(len(inducing)))
    if params['path_inducing_chp'] is not None:
        inducing.to_traj(params['path_inducing_chp'])
        log.write('path_inducing_chp: {}\n'.format(
            params['path_inducing_chp']))

    # Gaussian Process
    if params['path_gp_chp'] and os.path.isfile(params['path_gp_chp']):
        with open(params['path_gp_chp'], 'r') as f:
            gp = eval(f.readlines()[-1])
            kerns = gp.kern.kernels
        log.write('path_gp_chp: {} (read)\n'.format(params['path_gp_chp']))
    else:
        # numbers and pairs
        if params['numbers'] is None:
            _num = set()
            for atoms in data:
                for n in set(atoms.numbers):
                    _num.add(n)
            numbers = sorted(list(_num))
            params['numbers'] = numbers
        pairs = ([(a, b) for a, b in itertools.combinations(numbers, 2)] +
                 [(a, a) for a in numbers])
        log.write('numbers: {}\n'.format(params['numbers']))

        # kerns
        log.write('pairkernel: {}\nsoapkernel: {}\n'.format(
            params['pairkernel'], params['soapkernel']))
        kerns = []
        if params['pairkernel']:
            kerns += [PairKernel(RBF(), a, b, factor=PolyCut(params['cutoff']))
                      for a, b in pairs]
        if params['soapkernel']:
            kerns += [SoapKernel(Positive(1.0, requires_grad=True)*Normed(DotProd()**params['exponent']),
                                 atomic_number, params['numbers'], params['lmax'], params['nmax'],
                                 PolyCut(params['cutoff']), atomic_unit=params['atomic_unit'])
                      for atomic_number in params['numbers']]
            log.write('lmax: {}\n nmax: {}\nexponent: {}\natomic_unit: {}\n'.format(
                params['lmax'], params['nmax'], params['exponent'], params['atomic_unit']))

        gp = GaussianProcessPotential(
            kerns, noise=White(signal=params['error'], requires_grad=True))
        if params['path_gp_chp']:
            gp.to_file(params['path_gp_chp'], flag='initial state', mode='w')
            log.write('path_gp_chp: {} (write)\n'.format(
                params['path_gp_chp']))
    log.close()

    # train
    data.update(descriptors=kerns)
    inducing.stage(kerns)
    inducing.trainable = False  # TODO: this should be set inside Locals
    state = 0
    for steps in iterable(train):
        train_gpp(gp, data, inducing=inducing, steps=steps,
                  logprob_loss=True, cov_loss=False)
        # save gp
        state += steps
        if state > 0 and params['path_gp_chp']:
            gp.to_file(params['path_gp_chp'],
                       flag='state: {}'.format(state))
            with open('log.txt', 'a') as log:
                log.write('path_gp_chp: {} (write, state={})\n'.format(
                    params['path_gp_chp'], state))

        # save inducing
        if inducing.trainable:
            raise NotImplementedError(
                'trainable inducing is not implemented yet!')
    if state > 0:
        with open('log.txt', 'a') as log:
            log.write('\ntrained for {} steps\n'.format(state))

    # create Posterior Potential
    V = PosteriorPotential(gp, data, inducing)

    # test
    if params['test']:
        data.set_per_atoms('predicted_energy', V(data, 'energy'))
        data.set_per_atom('predicted_forces', V(data, 'forces'))
        var_e = data.target_energy.var()
        var_ee = (data.cat('predicted_energy') - data.target_energy).var()
        R2_e = 1-var_ee/var_e
        var_f = data.target_forces.var()
        var_ff = (data.cat('predicted_forces') - data.target_forces).var()
        R2_f = 1-var_ff/var_f
        with open('log.txt', 'a') as log:
            log.write('\ntesting the model on the same data that created it:')
            log.write('\nenergy R2 score={}'.format(R2_e))
            log.write('\nforces R2 score={}\n'.format(R2_f))

    return V

