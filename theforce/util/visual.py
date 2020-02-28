import pylab as plt
import nglview
from ase.io import read
from theforce.util.util import timestamp, iterable
import re


def no_preprocess(atoms):
    return atoms


def show_trajectory(traj, radiusScale=0.3, remove_ball_and_stick=False, preprocess=no_preprocess, sl=':'):
    if type(traj) == str:
        data = read(traj, sl)
    else:
        data = traj
    data = [preprocess(atoms) for atoms in iterable(data)]
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


def visualize_leapfrog(file, plot=True, extremum=False, stop=None):
    energies = []
    temperatures = []
    exact_energies = []
    acc = []
    undo = []
    ext = []
    times = []
    t0 = None
    for line in open(file):
        split = line.split()[2:]

        try:
            step = int(split[0])
        except IndexError:
            continue

        if stop and step > stop:
            break

        try:
            energies += [(step, float(split[1]))]
            temperatures += [(step, float(split[2]))]
            # time
            t = timestamp(' '.join(line.split()[:2]))
            if t0 is None:
                t0 = t
            times += [(step, t-t0)]
            t0 = t
        except:
            pass

        if 'a model is provided' in line:
            s = re.search('with (.+?) data and (.+?) ref', line)
            a = int(s.group(1))
            b = int(s.group(2))
            data = [(step, a)]
            refs = [(step, b)]
            fp = [(step, 0)]

        if 'updating ...' in line:
            updating = step

        if split[1] == 'undo:':
            assert int(split[2]) == exact_energies[-1][0]
            undo += [exact_energies[-1]]
            a, b = (int(_) for _ in split[4::2])
            data += [(step, data[-1][1]), (step, a)]
            refs += [(step, refs[-1][1]), (step, b)]

        if 'exact energy' in line:
            energy = float(split[3])
            exact_energies += [(step, energy)]
            acc += [None]

        if split[1] == 'update:':
            a, b, c = (int(_) for _ in split[4::2])
            try:
                data += [(step, data[-1][1]), (step, a)]
                refs += [(step, refs[-1][1]), (step, b)]
                fp += [(step, fp[-1][1]), (step, c)]
            except NameError:
                data = [(step, a)]
                refs = [(step, b)]
                fp = [(step, c)]

            if len(acc) > 0 and acc[-1] is None:
                acc[-1] = data[-1][1]
                if acc[-1] > 1:
                    acc[-1] -= data[-2][1]

        if 'extremum' in line:
            ext += [step]

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(8, 4))
        axes = axes.reshape(-1)

        #
        axes[0].plot(*zip(*energies), zorder=1)
        if len(exact_energies) > 0:
            axes[0].scatter(*zip(*exact_energies),
                            c=list(map({0: 'r', 1: 'g'}.get, acc)),
                            zorder=2)
            if len(undo) > 0:
                axes[0].scatter(*zip(*undo), marker='x', color='k', zorder=2)
        if extremum:
            for e in ext:
                axes[0].axvline(x=e, lw=0.5, color='k')
        axes[0].set_ylabel('energy')

        #
        axes[1].plot(*zip(*temperatures))
        axes[1].set_ylabel('temperature')

        #
        axes[2].plot(*zip(*data))
        axes[2].plot(*zip(*fp))
        axes[2].set_ylabel('FP calculations')

        #
        axes[3].plot(*zip(*refs))
        axes[3].set_ylabel('inducing')

        #
        for ax in axes:
            ax.set_xlim(0, step)
        fig.tight_layout()
    else:
        fig = None
    return energies, temperatures, exact_energies, data, refs, fp, fig, times

