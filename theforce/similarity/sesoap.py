# +
from theforce.descriptor.cutoff import PolyCut
from theforce.descriptor.sesoap import SeSoap, SubSeSoap
from theforce.regression.kernel import DotProd
from theforce.similarity.heterosoap import HeterogeneousSoapKernel
from theforce.similarity.universal import UniversalSoapKernel
from theforce.util.util import EqAll


class SeSoapKernel(UniversalSoapKernel):
    def __init__(self, lmax, nmax, exponent, cutoff, a=None, radii=1.0, normalize=True):
        radial = PolyCut(cutoff) if type(cutoff) == float else cutoff
        super().__init__(lmax, nmax, exponent, cutoff)
        self.descriptor = SeSoap(lmax, nmax, radial, radii=radii, normalize=normalize)
        self.dim = self.descriptor.dim
        self._a = EqAll() if a is None else a
        self._args = f"{lmax}, {nmax}, {exponent}, {cutoff}, a={a}"

    @property
    def state_args(self):
        return f"{self._args}, radii={self.descriptor.radii}, normalize={self.descriptor.normalize}"

    def call_descriptor(self, loc, grad):
        return self.descriptor(loc._r, loc._b, grad=grad)


class SubSeSoapKernel(HeterogeneousSoapKernel):
    def __init__(self, lmax, nmax, exponent, cutoff, a, b, radii=1.0, normalize=True):
        kern = DotProd() ** exponent
        radial = PolyCut(cutoff) if type(cutoff) == float else cutoff
        super().__init__(kern, a, b, lmax, nmax, radial)
        self.descriptor = SubSeSoap(
            lmax, nmax, radial, b, radii=radii, normalize=normalize
        )
        self.dim = self.descriptor.dim
        self._args = f"{lmax}, {nmax}, {exponent}, {cutoff}, {a}, {b}"

    @property
    def state_args(self):
        return f"{self._args}, radii={self.descriptor.radii}, normalize={self.descriptor.normalize}"

    def call_descriptor(self, loc, grad):
        return self.descriptor(loc._r, loc._b, grad=grad)


def test_SeSoapKernel():
    import numpy as np

    from theforce.descriptor.atoms import AtomsData, TorchAtoms, namethem
    from theforce.similarity.universal import DiracDeltaChemical, PolyCut

    soap = SeSoapKernel(2, 2, 4, 3.0)
    # soap = eval(soap.state)
    print(soap)
    namethem([soap])

    #
    cell = np.ones(3) * 10
    positions = (
        np.array(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            ]
        )
        + cell / 2
        + np.random.uniform(-0.1, 0.1, size=(7, 3))
    )
    b = TorchAtoms(
        positions=positions,
        numbers=3 * [10] + 3 * [18] + [10],
        cell=cell,
        pbc=True,
        cutoff=3.0,
        descriptors=[soap],
    )
    # make natoms different in a, b. P.S. add an isolated atom.
    _pos = np.concatenate([positions, [[0.0, 0.0, 0.0], [3.0, 5.0, 5.0]]])
    a = TorchAtoms(
        positions=_pos,
        numbers=2 * [10, 18, 10] + [18, 10, 18],
        cell=cell,
        pbc=True,
        cutoff=3.0,
        descriptors=[soap],
    )

    # left/right-grad
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    if 1:
        soap([a], [b]).backward()
        test_left = a.xyz.grad.allclose(soap.leftgrad(a, b).view(-1, 3))
        max_left = (a.xyz.grad - soap.leftgrad(a, b).view(-1, 3)).max()
        print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
        test_right = b.xyz.grad.allclose(soap.rightgrad(a, b).view(-1, 3))
        max_right = (b.xyz.grad - soap.rightgrad(a, b).view(-1, 3)).max()
        print("rightgrad: {} \t max diff: {}".format(test_right, max_right))

    #
    if 1:
        data = AtomsData([a, b])
        inducing = data.sample_locals(5)
        inducing.stage(descriptors=[soap])
        soap(data, data)
        soap(data, inducing)
        soap(inducing, inducing)


def test_SubSeSoapKernel():
    import numpy as np

    from theforce.descriptor.atoms import AtomsData, TorchAtoms, namethem
    from theforce.descriptor.cutoff import PolyCut
    from theforce.regression.kernel import DotProd, Normed, Positive
    from theforce.regression.stationary import RBF

    soap = SubSeSoapKernel(2, 2, 4, PolyCut(3.0), 10, (18, 10))
    namethem([soap])
    print(soap)

    # create atomic systems
    # Note that when one of the displacement vectors becomes is exactly along the z-axis
    # because of singularity some inconsistensies exist with autograd.
    # For this reason we add a small random number to positions, until that bug is fixed.
    cell = np.ones(3) * 10
    positions = (
        np.array(
            [
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, 0.0),
            ]
        )
        + cell / 2
        + np.random.uniform(-0.1, 0.1, size=(7, 3))
    )

    b = TorchAtoms(
        positions=positions,
        numbers=3 * [10] + 3 * [18] + [10],
        cell=cell,
        pbc=True,
        cutoff=3.0,
        descriptors=[soap],
    )
    # make natoms different in a, b. P.S. add an isolated atom.
    _pos = np.concatenate([positions, [[0.0, 0.0, 0.0], [3.0, 5.0, 5.0]]])
    a = TorchAtoms(
        positions=_pos,
        numbers=2 * [10, 18, 10] + [18, 10, 18],
        cell=cell,
        pbc=True,
        cutoff=3.0,
        descriptors=[soap],
    )

    # left/right-grad
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    soap([a], [b]).backward()
    test_left = a.xyz.grad.allclose(soap.leftgrad(a, b).view(-1, 3))
    max_left = (a.xyz.grad - soap.leftgrad(a, b).view(-1, 3)).max()
    print("leftgrad: {}  \t max diff: {}".format(test_left, max_left))
    test_right = b.xyz.grad.allclose(soap.rightgrad(a, b).view(-1, 3))
    max_right = (b.xyz.grad - soap.rightgrad(a, b).view(-1, 3)).max()
    print("rightgrad: {} \t max diff: {}".format(test_right, max_right))

    # gradgrad-left
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.leftgrad(a, b).view(-1, 3) * a.xyz).sum().backward()
    v1 = a.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.gradgrad(a, b) * a.xyz.view(-1)[:, None]).sum().backward()
    v2 = a.xyz.grad.data
    print("gradgrad-left: {}".format(v1.allclose(v2)))

    # gradgrad-right
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.rightgrad(a, b).view(-1, 3) * b.xyz).sum().backward()
    v1 = b.xyz.grad.data
    a.update(posgrad=True, forced=True)
    b.update(posgrad=True, forced=True)
    (soap.gradgrad(a, b) * b.xyz.view(-1)[None]).sum().backward()
    v2 = b.xyz.grad.data
    print("gradgradright: {}".format(v1.allclose(v2)))

    # gradgraddiag
    test_diag = soap.gradgrad(a, a).diag().allclose(soap.gradgraddiag(a))
    print("gradgraddiag: {}".format(test_diag))


if __name__ == "__main__":
    test_SeSoapKernel()
    test_SubSeSoapKernel()
