#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import datetime
import os


def iterable(a, ignore=None):
    if a.__class__ != ignore and hasattr(a, '__iter__'):
        return a
    else:
        return (a, )


def one_liner(x):
    return re.sub(' +', ' ', str(x).replace('\n', ''))


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

