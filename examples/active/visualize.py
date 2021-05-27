# +
from theforce.calculator.active import log_to_figure
import pylab as plt

fig = log_to_figure('active.log')
plt.pause(10)
fig.show()
