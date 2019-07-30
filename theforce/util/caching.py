
# coding: utf-8

# In[ ]:


import functools


class UID:
    c = -1

    def __init__(self, id=None):
        if id:
            self.id = id
        else:
            self.id = UID.c + 1
            UID.c += 1

    @staticmethod
    def ident(*args, forced=True):
        for o in args:
            # TODO: if hasattr(o, UID) but o.UID.__class__ != UID
            if forced or not hasattr(o, 'UID'):
                setattr(o, 'UID', UID())

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id

    def __call__(self):
        return self.id


def method_caching(method):
    """
    After decorating desired methods, caching can be turned on (off) 
    by obj.method_caching = True (False).
    Caveat: if the states of an argument changes between calls, 
    the user should assign a new UID manually (like obj.UID = UID()).
    """

    @functools.wraps(method)
    def cacher(self, *args):
        if hasattr(self, 'method_caching') and self.method_caching:
            # ident
            UID.ident(*args, forced=False)
            uid = tuple(a.UID() for a in args)
            # cache
            try:
                _return = self.cached[method.__name__][uid]
            except KeyError:
                _return = method(self, *args)
                if method.__name__ not in self.cached:
                    self.cached[method.__name__] = {}
                self.cached[method.__name__][uid] = _return
            except AttributeError:
                self.cached = {}
                _return = cacher(self, *args)  # recursive
            return _return
        else:
            return method(self, *args)

    return cacher


def example():
    import time

    class Feature:
        def __init__(self, n):
            self.n = n

    class Kern:

        def __init__(self, sleep=1.0):
            self.sleep = sleep

        def __call__(self, *args):
            return self.method(*args)

        @method_caching
        def method(self, a, b):
            time.sleep(self.sleep)
            return a.n*b.n

    X = [Feature(i) for i in range(2)]
    # UID.ident(X)
    k = Kern()
    k.method_caching = True  # or False (default) for no caching
    t1 = time.time()
    k(X[0], X[1])
    t2 = time.time()
    print('first evaluation: {} seconds'.format(t2-t1))
    k(X[0], X[1])
    t3 = time.time()
    print('second evaluation: {} seconds'.format(t3-t2))

    # clear cached
    k.cached.clear()


if __name__ == '__main__':
    example()

