# +
import importlib
import sys
import warnings

from ase.io import read

from theforce.util.server import Server
from theforce.util.util import date

_imported = {}


def reserve_ofile(o, msg="reserved"):
    with open(o, "w") as f:
        f.write(f"{date()} {msg}\n")


def get_calc(script, ref="calc"):
    scope = {}
    try:
        exec(open(script).read(), scope)
    except TypeError:
        exec(script.read(), scope)
    return scope[ref]


def _get_scope(script):
    scope = {}
    try:
        exec(open(script).read(), scope)
    except TypeError:
        exec(script.read(), scope)
    return scope


def get_scope(script):
    if script not in _imported:
        spec = importlib.util.spec_from_file_location("_import", script)
        _import = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_import)
        _imported[script] = _import
    _module = _imported[script]
    scope = {"calc": _module.calc}
    try:
        scope["preprocess_atoms"] = _module.preprocess_atoms
    except:
        pass
    try:
        scope["postprocess_atoms"] = _module.postprocess_atoms
    except:
        pass
    return scope


def calculate(file, calc):
    scope = {}
    if ":" in file:
        msg = file.split(":")
        if len(msg) == 2:
            i, o = msg
        elif len(msg) == 3:
            i, o, c = msg
            scope = get_scope(c)
            calc = scope["calc"]
        elif len(msg) == 4:
            i, o, c, ref = msg
            calc = get_scope(c)
            calc = scope[ref]
        else:
            raise RuntimeError(f"message > 3 -> {msg}")
    else:
        i = o = file
    try:
        reserve_ofile(o)
        atoms = read(i)
        atoms.calc = calc
        if "preprocess_atoms" in scope:
            scope["preprocess_atoms"](atoms)
        atoms.get_potential_energy()
        atoms.get_forces()
        atoms.get_stress()
        if "postprocess_atoms" in scope:
            scope["postprocess_atoms"](atoms)
        atoms.write(o, format="extxyz")
    except FileNotFoundError:
        warnings.warn(f"unable to read {i} -> calculation skipped")


if __name__ == "__main__":
    import argparse

    from theforce.util.ssh import clear_port

    parser = argparse.ArgumentParser(description="Starts a calculation server.")
    parser.add_argument("-ip", "--ip", default="localhost")
    parser.add_argument("-port", "--port", type=int, default=6666)
    parser.add_argument(
        "-calc",
        "--calculator",
        default=None,
        help=(
            "If given, it should be a python script in which "
            + "a variable named calc is defined."
        ),
    )
    args = parser.parse_args()

    if args.calculator is not None:
        calc = get_calc(args.calculator)
    else:
        calc = None

    clear_port(args.port)
    s = Server(args.ip, args.port, callback=calculate, args=(calc,))
    s.listen()
