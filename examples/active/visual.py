# +
import pylab as plt

from theforce.calculator.active import log_to_figure

fig = log_to_figure("active.log")
plt.pause(10)
fig.show()
