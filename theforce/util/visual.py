# +
import pylab as plt
import nglview
from ase.io import read
from theforce.util.util import timestamp, iterable
import re
from torch import tensor


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


def visualize_leapfrog(file, plot=True, extremum=False, stop=None, mlcolor=None, colors=None):
    energies = []
    temperatures = []
    exact_energies = []
    ml_energies = []
    ediff = []
    stats = []
    acc = []
    undo = []
    ext = []
    times = []
    t0 = None
    progress = None
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
            a = line.split()
            try:
                progress = a[5]
            except IndexError:
                pass

        if 'ediff at break' in line:
            e = line.split()[-1]
            if 'tensor' in e:
                e = float(eval(e))
            ediff += [(step, float(e))]

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
            if step > 0:
                ml_energies += [energies[-1]]

        if 'stats' in line:
            stats += [[step] + [float(v) for v in line.split()[-4:]]]

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
        if colors:
            color_a, color_b = colors
        else:
            color_a, color_b = 'chocolate', 'cornflowerblue'

        # ---------------------
        axes[0].plot(*zip(*energies), zorder=1, color=color_a)
        axes[0].set_ylabel('energy', color=color_a)
        axes[0].tick_params(axis='y', colors=color_a)
        if len(exact_energies) > 0:
            axes[0].scatter(*zip(*exact_energies),
                            c=list(map({0: 'r', 1: 'g'}.get, acc)),
                            zorder=2)
            if mlcolor:
                axes[0].scatter(*zip(*ml_energies),  color=mlcolor)
            if len(undo) > 0:
                axes[0].scatter(*zip(*undo), marker='x', color='k', zorder=2)
        if extremum:
            for e in ext:
                axes[0].axvline(x=e, lw=0.5, color='k')

        #
        ax = axes[0].twinx()
        ax.plot(*zip(*temperatures), color=color_b, alpha=0.7)
        ax.set_ylabel('temperature', color=color_b)
        ax.tick_params(axis='y', colors=color_b)

        # ---------------------
        if len(stats) > 0:
            t, e, ee, f, fe = zip(*stats)
            ax = axes[1]
            ax.errorbar(t, f, yerr=fe, capsize=5, fmt='.', color=color_a)
            ax.set_ylabel('fdiff', color=color_a)
            ax.tick_params(axis='y', colors=color_a)
            #
            ax = axes[1].twinx()
            ax.errorbar(t, e, yerr=ee, capsize=5, fmt='.', color=color_b)
            ax.set_ylabel('ediff', color=color_b)
            ax.tick_params(axis='y', colors=color_b)

        # ---------------------
        axes[2].plot(*zip(*data), color=color_a)
        axes[2].set_ylabel('data', color=color_a)
        axes[2].tick_params(axis='y', colors=color_a)
        axes[2].set_ylim(data[0][1], data[0][1]+fp[-1][1] + 3)
        ax = axes[2].twinx()
        ax.plot(*zip(*fp), color=color_b)
        ax.set_ylabel('FP calculations', color=color_b)
        ax.tick_params(axis='y', colors=color_b)
        ax.set_ylim(fp[0][1], fp[-1][1] + 3)

        # ---------------------
        axes[3].plot(*zip(*refs), color=color_a)
        axes[3].set_ylabel('inducing', color=color_a)
        axes[3].tick_params(axis='y', colors=color_a)
        if len(ediff) > 0:
            ax = axes[3].twinx()
            ax.scatter(*zip(*ediff), color=color_b)
            ax.set_ylim(0,)
            ax.set_ylabel('ediff at break', color=color_b)
            ax.tick_params(axis='y', colors=color_b)

        #
        for ax in axes:
            ax.set_xlim(0, step)
        fig.tight_layout()

        #
        if progress:
            from matplotlib.patches import Rectangle as Rect
            fig.subplots_adjust(top=0.85)
            ax = fig.add_axes([0.3, 0.9, 0.4, 0.07])
            fig.text(0.7, 0.925, progress)
            ax.axis('off')
            a, b = (int(v) for v in progress.split('/'))
            wy = b*0.05
            ax.set_xlim(-wy, b+wy)
            ax.set_ylim(-0.2, 1.2)
            ax.add_patch(Rect((0, 0), b, 1, fc=(0, 0, 1, 0.1)))
            ax.add_patch(Rect((0, 0), a, 1, fc=(0, 0, 1, 1)))
            ax.set_xlabel(progress)

    else:
        fig = None
    return energies, temperatures, exact_energies, data, refs, fp, fig, times
