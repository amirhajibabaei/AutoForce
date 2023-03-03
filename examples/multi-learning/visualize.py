# +
import pylab as plt

from theforce.calculator.active import log_to_figure

fig = log_to_figure("active_1.log")
fig.savefig("calc_1.pdf")
fig = log_to_figure("active_2.log")
fig.savefig("calc_2.pdf")
