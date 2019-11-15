#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import datetime
import os
import psutil


def iterable(a, ignore=None):
    if a.__class__ != ignore and hasattr(a, '__iter__'):
        return a
    else:
        return (a, )


def one_liner(x):
    return re.sub(' +', ' ', str(x).replace('\n', ''))


def date(fmt="%m/%d/%Y %H:%M:%S"):
    return datetime.datetime.now().strftime(fmt)


def timestamp(string=None, fmt="%m/%d/%Y %H:%M:%S"):
    if string is not None:
        d = datetime.datetime.strptime(string, fmt)
    else:
        d = datetime.datetime.now()
    return d.timestamp()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def safe_dirname(d, append='x'):
    dd = d
    if dd.endswith('/'):
        dd = dd[:-1]
    while os.path.isdir(dd):
        dd += append
    return dd


def meminfo():
    return os.getpid(), psutil.Process(os.getpid()).memory_info().rss

