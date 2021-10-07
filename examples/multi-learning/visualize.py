# +
from theforce.calculator.active import log_to_figure
import pylab as plt

fig = log_to_figure('active_1.log')
fig.savefig('calc_1.pdf')
fig = log_to_figure('active_2.log')
fig.savefig('calc_2.pdf')
