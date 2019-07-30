
# coding: utf-8

# In[ ]:


import re


def iterable(a, ignore=None):
    if a.__class__ != ignore and hasattr(a, '__iter__'):
        return a
    else:
        return (a, )


def one_liner(x):
    return re.sub(' +', ' ', str(x).replace('\n', ''))

