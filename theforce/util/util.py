
# coding: utf-8

# In[ ]:


def iterable(a):
    if hasattr(a, '__iter__'):
        return a
    else:
        return (a, )

