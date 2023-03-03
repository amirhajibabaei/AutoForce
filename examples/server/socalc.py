from ase.build import bulk

from theforce.calculator.socketcalc import SocketCalculator

calc = SocketCalculator(script="calc.py")
atoms = bulk("Au")
atoms.set_calculator(calc)
e = atoms.get_potential_energy()
print(f"potential energy = {e}")
