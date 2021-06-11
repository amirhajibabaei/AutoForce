import itertools
import os
import random
import numpy as np
import torch


def get_params(**kwargs):

    # default params
    params = {
        # kernel
        'numbers': None,
        'cutoff': None,
        'atomic_unit': None,
        'lmax': 2,
        'nmax': 2,
        'exponent': 4,
        'pairkernel': True,
        'soapkernel': True,
        'heterosoap': True,
        'noise': 0.01,
        'noisegrad': True,
        # data
        'path_data': None,
        'path_data_chp': None,
        'path_inducing_chp': None,
        'use_gp_chp': True,
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


def get_kernel(params):
    from theforce.regression.gppotential import GaussianProcessPotential
    from theforce.similarity.pair import PairKernel
    from theforce.similarity.soap import SoapKernel, NormedSoapKernel
    from theforce.similarity.heterosoap import HeterogeneousSoapKernel
    from theforce.regression.stationary import RBF
    from theforce.descriptor.cutoff import PolyCut
    from theforce.regression.kernel import White, Positive, DotProd, Normed, Mul, Pow, Add
    from torch import tensor

    # Gaussian Process
    if params['path_gp_chp'] and params['use_gp_chp'] and os.path.isfile(params['path_gp_chp']):
        with open(params['path_gp_chp'], 'r') as f:
            gp = eval(f.readlines()[-1])
            kerns = gp.kern.kernels

        # log
        if params['path_log']:
            with open(params['path_log'], 'a') as log:
                log.write('path_gp_chp: {} (read)\n'.format(
                    params['path_gp_chp']))
    else:

        # log
        if params['path_log']:
            with open(params['path_log'], 'a') as log:
                log.write('pairkernel: {}\nsoapkernel: {}\n'.format(
                    params['pairkernel'], params['soapkernel']))

        kerns = []
        if params['pairkernel']:
            pairs = ([(a, b) for a, b in itertools.combinations(params['numbers'], 2)] +
                     [(a, a) for a in params['numbers']])
            kerns += [PairKernel(RBF(), a, b, factor=PolyCut(params['cutoff']))
                      for a, b in pairs]
        if params['soapkernel']:
            if params['heterosoap']:
                SOAP = HeterogeneousSoapKernel
            else:
                SOAP = NormedSoapKernel
            kerns += [SOAP(Positive(1.0, requires_grad=True)*DotProd()**params['exponent'],
                           atomic_number, params['numbers'], params['lmax'], params['nmax'],
                           PolyCut(params['cutoff']), atomic_unit=params['atomic_unit'])
                      for atomic_number in params['numbers']]
            # log
            if params['path_log']:
                with open(params['path_log'], 'a') as log:
                    log.write('lmax: {}\n nmax: {}\nexponent: {}\natomic_unit: {}\n'.format(
                        params['lmax'], params['nmax'], params['exponent'], params['atomic_unit']))

        gp = GaussianProcessPotential(
            kerns, noise=White(signal=params['noise'], requires_grad=params['noisegrad']))

        if params['path_gp_chp']:
            gp.to_file(params['path_gp_chp'], flag='created', mode='w')

            # log
            if params['path_log']:
                with open(params['path_log'], 'a') as log:
                    log.write('path_gp_chp: {} (write)\n'.format(
                        params['path_gp_chp']))

    return gp


def kernel_from_state(state):
    from theforce.regression.gppotential import GaussianProcessPotential
    from theforce.similarity.pair import PairKernel
    from theforce.similarity.soap import SoapKernel
    from theforce.regression.stationary import RBF
    from theforce.descriptor.cutoff import PolyCut
    from theforce.regression.kernel import White, Positive, DotProd, Normed, Mul, Pow, Add
    from torch import tensor
    return eval(state)


def potential_energy_surface(data=None, inducing=None, train=0, caching=False, append_log=True, **kwargs):
    from theforce.descriptor.atoms import AtomsData, LocalsData, sample_atoms
    from theforce.regression.gppotential import PosteriorPotential
    from theforce.regression.gppotential import train_gpp
    from theforce.util.util import iterable

    # get params
    params = get_params(**kwargs)
    log = open(params['path_log'], 'a' if append_log else 'w')
    log.write('{} threads: {}\n'.format(37*'*', torch.get_num_threads()))

    # data
    if data is None:
        data = sample_atoms(params['path_data'], size=params['ndata'],
                            chp=params['path_data_chp'])
        data.update(cutoff=params['cutoff'])
        natoms = sum([len(atoms) for atoms in data])
        log.write('cutoff: {}\npath_data: {}\nndata: {} (={} locals)\npath_data_chp: {}\n'.format(
            params['cutoff'], params['path_data'], params['ndata'], natoms, params['path_data_chp']))
    else:
        # if data is given as kwarg, it already should be cutoff-ed.
        assert len(data[-1]) == data[-1].natoms
        natoms = sum([len(atoms) for atoms in data])
        log.write('ndata: {} (={} locals) (kwarg)\n'.format(len(data), natoms))

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

    # numbers
    if params['numbers'] is None:
        params['numbers'] = data.numbers_set()
        log.write('numbers: {}\n'.format(params['numbers']))
    log.close()

    # kernel
    gp = get_kernel(params)

    # train
    data.update(descriptors=gp.kern.kernels)
    inducing.stage(gp.kern.kernels)
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
    V = PosteriorPotential(gp, data, inducing=inducing, use_caching=caching)

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
        print('testing the model on the same data that created it:')
        print('energy R2 score={}'.format(R2_e))
        print('forces R2 score={}'.format(R2_f))

    return V

