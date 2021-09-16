# +
from ase.calculators.calculator import Calculator, all_changes
import numpy as np


# Only for quick tests!
#
class ZeroCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy', 'stress']

    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results['energy'] = 0.
        self.results['forces'] = np.zeros([len(self.atoms), 3])
        self.results['free_energy'] = 0.
        self.results['stress'] = np.zeros(6)


calc = ZeroCalculator()
