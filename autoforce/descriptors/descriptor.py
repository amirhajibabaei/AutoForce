class Descriptor:

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
