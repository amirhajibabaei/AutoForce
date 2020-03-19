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
