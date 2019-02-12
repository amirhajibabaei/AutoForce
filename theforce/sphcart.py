
# coding: utf-8

# In[1]:


import numpy as np


def cart_to_angles( x, y, z ):
    rxy_sq = x*x + y*y
    r      = np.sqrt( rxy_sq + z*z )
    theta  = np.arctan2( np.sqrt(rxy_sq), z )
    phi    = np.arctan2( y, x )
    return r, theta, phi


def cart_to_sph( self, x, y, z ):
    rxy_sq    = x*x + y*y 
    rxy       = np.sqrt( rxy_sq ) + self.tiny
    r_sq      = rxy_sq + z*z
    r         = np.sqrt( r_sq )
    sin_theta = rxy/r
    cos_theta = z/r
    sin_phi   = y/rxy
    cos_phi   = x/rxy
    return r, sin_theta, cos_theta, sin_phi, cos_phi


def angles_to_cart( r, theta, phi ):
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return x, y ,z


def sph_to_cart( sin_theta, cos_theta, sin_phi, cos_phi, F_r, F_theta, F_phi ):
    F_x = sin_theta * cos_phi * F_r + cos_theta * cos_phi * F_theta - sin_phi * F_phi
    F_y = sin_theta * sin_phi * F_r + cos_theta * sin_phi * F_theta + cos_phi * F_phi
    F_z = cos_theta * F_r - sin_theta * F_theta
    return F_x, F_y, F_z

