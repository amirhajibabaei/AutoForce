class Descriptor:

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def name(self):
        return self.__class__.__name__
