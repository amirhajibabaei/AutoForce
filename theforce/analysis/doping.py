# +
import numpy as np


def normalized_formula(formula):
    total = sum(formula.values())
    return {s: c/total for s, c in formula.items()}


def sign(a):
    if a == 0:
        return 0
    elif a > 0:
        return 1
    else:
        return -1


def error_function(_a, _b):
    """
    error includes a max term plus mean of weighted differences
    """
    species = set(_a.keys()).union(set(_b.keys()))
    a = normalized_formula(_a)
    b = normalized_formula(_b)
    x = np.array([a[s] for s in species])
    y = np.array([b[s] for s in species])
    rho = (x + y)/2
    diff = abs(x-y)
    err = diff.max() + (diff*np.exp(-rho)).mean()
    return err


def configure_doping(prim, target, mul=(1, 2, 3, 4, 6)):
    """
    return the best solution:
       repeat, initial, solution, delta, errors
    """

    numbers = {s: c for s, c in zip(*np.unique(prim.numbers, return_counts=1))}
    species = set(numbers.keys()).union(set(target.keys()))

    def opt(mul):
        initial = {}
        for s in species:
            initial[s] = numbers[s]*mul if s in numbers else 0
            if s not in target:
                target[s] = 0
        n = sum(initial.values())
        tar = normalized_formula(target)
        ini = normalized_formula(initial)
        delta = {s: int(round((tar[s]-ini[s])*n)) for s in species}
        sol = {s: initial[s] + delta[s] for s in species}
        for s in species:
            if sol[s] < 0:
                delta[s] -= sol[s]
                sol[s] = 0
        res = sum(delta.values())
        while res != 0:
            ae = 1
            best = None
            d = -sign(res)
            for s in species:
                if sol[s] + d > 0:
                    sol[s] += d
                    e = error_function(sol, target)
                    if e < ae:
                        best = s
                        ae = e
                    sol[s] -= d
            sol[best] += d
            delta[best] += d
            res = sum(delta.values())

        assert [sol[s] > 0 for s in species]
        assert [sol[s] == initial[s]+delta[s] for s in species]
        err = error_function(sol, target)

        return initial, sol, delta, err

    # iterate over repeatitions
    ae = 1
    errors = {}
    for m in sorted(list(mul)):
        tmp = opt(m)
        errors[m] = tmp[3]
        if tmp[3] < ae:
            ae = errors[m]
            best = tmp
            repeat = m

    initial = best[0]
    solution = best[1]
    delta = best[2]
    return repeat, initial, solution, delta, errors


def random_doping(atoms, delta, mask=None):
    to = []
    subs = []
    if mask is None:
        mask = len(atoms)*[True]
    for a, b in delta.items():
        if b > 0:
            to += b*[a]
        elif b < 0:
            cand = [at.index for at in atoms if
                    (at.number == a and mask[at.index]
                     and at.index not in subs)]
            subs += np.random.choice(cand, abs(b), replace=False).tolist()
    subs = np.random.permutation(subs).tolist()
    doped = atoms.copy()
    doped.numbers[subs] = to
    return doped, subs, to
