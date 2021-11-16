# +
from torch import Tensor


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
