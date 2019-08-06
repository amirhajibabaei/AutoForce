#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from theforce.regression.scores import coeff_of_determination
from theforce.calculator.parametric import get_coulomb_terms, get_lj_terms, load_parametric_potential
import os


def train_parametric_potential(data, cutoff=None, paramed_pot=None, chp='paramedpot.chp', use_chp=True,
                               charges_setting={}, threshold=None, max_steps=100):

    # read
    if paramed_pot == None and os.path.isfile(chp) and use_chp:
        with open(chp) as file:
            paramed_pot = load_parametric_potential(file.readlines()[-1])
            print('paramed_pot read from {}'.format(chp))

    # create
    if paramed_pot is None:
        numbers = data.numbers_set()
        paramed_pot = (get_coulomb_terms(numbers, cutoff, setting=charges_setting)
                       + get_lj_terms(numbers, cutoff))
        paramed_pot.to_file(chp, flag='created')
        print('paramed_pot created')

    # threshold: train only on forces larger than a threshold
    target = data.cat('target_forces')
    f = target.pow(2).sum(dim=1).sqrt()
    if threshold is None:
        threshold = f.var().sqrt()
    m = f > threshold

    # training setup
    def forces_loss(a, b):
        return (a-b).pow(2).sum()
    optimizer = torch.optim.RMSprop(paramed_pot.unique_params, lr=0.1)

    # train
    losses = []
    for _ in range(max_steps):
        # step
        optimizer.zero_grad()
        pred = torch.cat([paramed_pot(atoms, forces=True)[1]
                          for atoms in data])
        loss = forces_loss(pred[m], target[m])
        loss.backward()
        optimizer.step()

        cod_full = coeff_of_determination(pred, target)
        cod_thresh = coeff_of_determination(pred[m], target[m]).data
        print('{} loss: {}     R2 (full): {}     R2 (>thresh): {}'.format(
            _, loss.data, cod_full.data, cod_thresh))

        losses += [loss.data]
        paramed_pot.to_file(chp, flag='step: {}   r2 on all forces: {}   r2 on large forces = {}'.format(
            _, cod_full.data, cod_thresh.data))

    return paramed_pot

