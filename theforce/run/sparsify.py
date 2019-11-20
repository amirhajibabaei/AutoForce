#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from theforce.regression.algebra import sparser_projection
from theforce.regression.gppotential import PosteriorPotentialFromFolder
import os


def sparsify_saved_model(input_model, alpha=1.0, sweeps=10, output_model=None, report=True, plot=True):

    if output_model is None:
        output_model = os.path.join(os.path.dirname(input_model),
                                    os.path.basename(input_model))+'_sparse'
        j = 0
        suffix = ''
        while os.path.isdir(output_model+suffix):
            j += 1
            suffix = '_{}'.format(j)
        output_model += suffix

    V = PosteriorPotentialFromFolder(
        input_model, load_data=True, update_data=False)
    M = V.M
    K = V.K
    D = V.gp.diagonal_ridge(V.data)
    Y = V.gp.Y(V.data)
    indices = None
    deleted = None
    K, M, indices, deleted = sparser_projection(K, M, Y, D, alpha=alpha, sweeps=sweeps,
                                                indices=indices, deleted=deleted)
    V_M = V.M
    V_K = V.K
    V_mu = V.mu
    V.select_inducing(indices, deleted=deleted)
    V.to_folder(output_model)

    if report:
        import torch
        with open(os.path.join(output_model, '_sparse'), 'w') as report:
            report.write('input model: {}\n'.format(input_model))
            i = torch.tensor(indices)
            report.write('test: {}\n'.format(
                V_M.index_select(0, i).index_select(1, i).allclose(M)))
            report.write('{} + {} = {} ? {}\n'.format(len(deleted), len(i),
                                                      len(deleted)+len(i), V_M.size(0)))
            report.write('deleted references:\n{}\n'.format(deleted))

    if plot:
        from theforce.regression.algebra import projected_process_auxiliary_matrices_D
        import pylab as plt
        # %matplotlib inline

        mu, nu, ridge, choli = projected_process_auxiliary_matrices_D(
            K, M, Y, D, chol_inverse=True)
        _ = plt.hist((V_K@V_mu-Y).detach(), bins=100,
                     color='cyan', alpha=0.5, label='full')
        _ = plt.hist((K@mu-Y).detach(), bins=100,
                     color='green', alpha=0.5, label='sparse')
        plt.legend()
        plt.savefig(os.path.join(output_model, '_sparse.eps'))
        plt.savefig(os.path.join(output_model, '_sparse.png'))

