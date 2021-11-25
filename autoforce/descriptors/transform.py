# +
import torch
from torch import Tensor
from typing import Union, Sequence


def r_theta_phi(v: Tensor) -> (Tensor, Tensor, Tensor):
    """
    v: (n, 3)-shaped tensor of Cartesian coordinates

    returns: r(n), theta(n), phi(n) of spherical coordinates

    """

    x, y, z = v.T
    rxy2 = x*x + y*y
    rxy = rxy2.sqrt()
    r = (rxy2 + z*z).sqrt()
    theta = torch.atan2(rxy, z)
    phi = torch.atan2(y, x)

    return r, theta, phi


def cartesian(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Inverse of r_theta_phi function.

    """

    x = r*theta.sin()*phi.cos()
    y = r*theta.sin()*phi.sin()
    z = r*theta.cos()

    return torch.stack([x, y, z]).T


def rotation_matrix(axis: Union[torch.Tensor, Sequence[float]],
                    theta: Union[torch.Tensor, float]
                    ) -> torch.Tensor:
    """
    axis: rotation axis (length 3)

    theta: rotation angle in Radians

    returns: a (3, 3)-shaped rotation matrix

    """

    axis = torch.as_tensor(axis)
    axis = axis/axis.norm()
    a = torch.as_tensor(theta/2).cos()
    b, c, d = -axis*torch.as_tensor(theta/2).sin()
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = torch.tensor([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

    return rot


def spherical_vector_to_cartesian(sin_theta: Tensor,
                                  cos_theta: Tensor,
                                  sin_phi: Tensor,
                                  cos_phi: Tensor,
                                  v_r: Tensor,
                                  v_theta: Tensor,
                                  v_phi: Tensor
                                  ) -> (Tensor, Tensor, Tensor):
    """
    Transforms a vector from spherical coordinates:
        (v_r, v_theta, v_phi)
    to Cartesian:
        (v_x, v_y, v_z)
    Instead of polar theta and azimuthal phi angles,
    their sin and cos values are used for transformation.

    """

    v_x = sin_theta*cos_phi*v_r + cos_theta*cos_phi*v_theta - sin_phi*v_phi
    v_y = sin_theta*sin_phi*v_r + cos_theta*sin_phi*v_theta + cos_phi*v_phi
    v_z = cos_theta*v_r - sin_theta*v_theta

    return v_x, v_y, v_z
