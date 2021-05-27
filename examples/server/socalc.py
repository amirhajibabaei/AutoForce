from theforce.calculator.socketcalc import SocketCalculator
from ase.build import bulk


calc = SocketCalculator(script='calc.py')
atoms = bulk('Au')
atoms.set_calculator(calc)
e = atoms.get_potential_energy()
print(f'potential energy = {e}')
