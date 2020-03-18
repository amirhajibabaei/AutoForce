from theforce.regression.gppotential import PosteriorPotentialFromFolder
from theforce.calculator.posterior import AutoForceCalculator
import torch.distributed as dist


def extremum(e):
    ext = False
    if len(e) >= 3:
        d1 = e[-1] - e[-2]
        d2 = e[-2] - e[-3]
        if d1*d2 < 0:
            ext = True
    return ext


def run_updates(dyn, steps):
    n = 0
    e = []
    while n < steps:
        dyn.run(1)
        e += [dyn.atoms.get_potential_energy()]
        if extremum(e):
            n += 1


def get_dyn(init):
    scope = {}
    try:
        exec(open(init).read(), scope)
    except TypeError:
        exec(init.read(), scope)
    return scope['dyn']


def run_dyn(model, init, steps):
    dyn = get_dyn(init)
    dist.init_process_group('mpi')
    calc = AutoForceCalculator(PosteriorPotentialFromFolder(model, load_data=False),
                               process_group=dist.group.WORLD)
    print(f'world: {dist.get_rank()}, {dist.get_world_size()}')
    dyn.atoms.set_calculator(calc)
    if steps > 0:
        dyn.run(steps)
    elif steps < 0:
        run_updates(dyn, abs(steps))


if __name__ == '__main__':
    import sys
    model = sys.argv[1]
    init = sys.argv[2]
    steps = int(sys.argv[3])
    run_dyn(model, init, steps)
