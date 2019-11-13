#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pylab as plt
import nglview
from ase.io import read


def show_trajectory(traj, radiusScale=0.3, remove_ball_and_stick=False):
    if type(traj) == str:
        data = read(traj, ':')
    else:
        data = traj
    view = nglview.show_asetraj(data)
    view.add_unitcell()
    view.add_spacefill()
    if remove_ball_and_stick:
        view.remove_ball_and_stick()
    view.camera = 'orthographic'
    view.parameters = {"clipDist": 0}
    view.center()
    view.update_spacefill(radiusType='covalent',
                          radiusScale=radiusScale,
                          color_scale='rainbow')
    return view


def visualize_leapfrog(file, plot=True):
    energies = []
    temperatures = []
    exact_energies = []
    data = []
    refs = []
    fp = []
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
            energy = float(split[3])
            exact_energies += [(step, energy)]

        try:
            if split[1] == 'update:':
                a, b, c = (int(_) for _ in split[4::2])
                data += [(step, a)]
                refs += [(step, b)]
                fp += [(step, c)]
        except IndexError:
            pass

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        axes = axes.reshape(-1)

        axes[0].plot(*zip(*energies), zorder=1)
        if len(exact_energies) > 0:
            axes[0].scatter(*zip(*exact_energies), color='red', zorder=2)

        axes[1].plot(*zip(*temperatures))

        axes[2].plot(*zip(*data))
        axes[2].plot(*zip(*fp))

        axes[3].plot(*zip(*refs))
        fig.tight_layout()
    else:
        fig = None
    return energies, temperatures, exact_energies, data, refs, fp, fig

