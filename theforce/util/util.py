# +
import re
import datetime
import os
import psutil


def iterable(a, ignore=None):
    if a.__class__ != ignore and hasattr(a, '__iter__'):
        return a
    else:
        return (a, )


def iter_balanced(it1, it2):
    less = min(len(it1), len(it2))
    more = max(len(it1), len(it2))
    left = len(it1) == less
    a = more//less
    b = more % less
    chunks = [a+1 if j < b else a for j in range(less)]
    # iterate
    start = 0
    for i, chunk in enumerate(chunks):
        if left:
            yield [it1[i]], [it2[j] for j in range(start, start+chunk)]
        else:
            yield [it1[j] for j in range(start, start+chunk)], [it2[i]]
        start = start+chunk


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


def class_of_method(method):
    # answered by estani at:
    # https://stackoverflow.com/questions/961048/get-class-that-defined-method
    method_name = method.__name__
    if method.__self__:
        classes = [method.__self__.__class__]
    else:
        # unbound method
        classes = [method.im_class]
    while classes:
        c = classes.pop()
        if method_name in c.__dict__:
            return c
        else:
            classes = list(c.__bases__) + classes
    return None


def rounded(_p, s=2):
    p = float(_p)
    c = 1
    while abs(c*p) < 1.:
        c *= 10
    return round(c*p, s)/c
