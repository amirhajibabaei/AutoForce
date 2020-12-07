# +
# show training
from theforce.calculator.active import log_to_figure

fig = log_to_figure('active.log')

# +
# visualize trajectory using nglview
from theforce.util import visual

visual.show_trajectory('md.traj')

# +
# mean squared displacement of atom 0
from theforce.util.analysis import TrajAnalyser
import pylab as plt

ana = TrajAnalyser('md.traj')
plt.plot(*ana.msd(I=[0]))
