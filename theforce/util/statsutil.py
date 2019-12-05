#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


def moving_average(x, w, axis=0):
    a = np.cumsum(x, dtype=np.float, axis=axis)
    a[w:] = a[w:] - a[:-w]
    return a[w-1:] / w

