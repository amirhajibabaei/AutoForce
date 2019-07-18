
# coding: utf-8

# In[ ]:


def iterable(a, ignore=None):
    if a.__class__ != ignore and hasattr(a, '__iter__'):
        return a
    else:
        return (a, )

