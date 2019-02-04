
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import pi


class sph_repr:
    
    def __init__( self, lmax, tiny=1.0e-16 ):
        
        self.lmax = lmax
        self.tiny = tiny
        self.lmax_p = lmax+1

        # pre-calculate 
        self.Yoo    = np.sqrt( 1./(4*pi) )
        self.alp_al = 2*[[]] + [ np.array( [ np.sqrt( (4.*l*l-1.)/(l*l-m*m) ) 
                                for m in range(l-1)][::-1] ) [:,np.newaxis]
                                for l in range(2,self.lmax_p) ]
        self.alp_bl = 2*[[]] + [ np.array( [ 
                                -np.sqrt( ((l-1.)**2-m*m)/(4*(l-1.)**2-1) ) 
                                for m in range(l-1)][::-1] ) [:,np.newaxis]
                                for l in range(2,self.lmax_p) ]
        self.alp_cl = [ np.sqrt(2.*l+1.) for l in range(self.lmax_p) ] 
        self.alp_dl = [[]] + [ -np.sqrt(1.+1./(2.*l)) 
                                for l in range(1,self.lmax_p) ]
        
        # indices: for traversing diagonals
        self.I = [[ l+k for l in range(lmax-k+1)] for k in range(lmax+1) ] 
        self.J = [[ l   for l in range(lmax-k+1)] for k in range(lmax+1) ] 
        
        # l,m tables
        self.l = np.array( [ [l for m in range(l)] + 
                             [m for m in range(l,self.lmax+1)] 
                                for l in range(self.lmax+1)] )[:,:,np.newaxis]
        self.m = np.empty_like(self.l)
        for m in range(self.lmax+1):
            self.m[ self.I[m], self.J[m] ] = m
            self.m[ self.J[m], self.I[m] ] = m
        
        # lower triangle indices
        self.tril_indices = np.tril_indices( self.lmax+1, k = -1 )
        
        # l,m related coeffs
        self.coef = np.sqrt( (self.l-self.m) * (self.l+self.m) 
                            * (2*self.l+1) / (2*self.l-1.) )[1:,1:]

        
        
    def cart_to_angles( self, x, y, z ):
        rxy_sq = x*x + y*y
        r      = np.sqrt( rxy_sq + z*z )
        theta  = np.arctan2( np.sqrt(rxy_sq), z )
        phi    = np.arctan2( y, x )
        return r, theta, phi
    
    
    
    def cart_to_sph( self, x, y, z ):
        rxy_sq    = np.atleast_1d( x*x + y*y )
        rxy       = np.sqrt( rxy_sq ) + self.tiny
        r_sq      = rxy_sq + z*z
        r         = np.sqrt( r_sq )
        sin_theta = rxy/r
        cos_theta = z/r
        sin_phi   = y/rxy
        cos_phi   = x/rxy
        return r, sin_theta, cos_theta, sin_phi, cos_phi


    
    def ylm( self, x, y, z ):
        """ 
        Inputs: x, y, z Cartesian coordinates
        Returns: r, sin_theta, cos_theta, sin_phi, cos_phi, Y
        r: radius, shape is like x
        sin_theta, cos_theta, sin_phi, cos_phi: sin and cos of theta, phi
        Y: spherical harmonics, shape = (lmax+1,lmax+1,*np.shape(x))
        
        ------------------------------------------------------------------------
        
        The imaginary componenst are stored in the upper diagonal of array Y.
        l = 0,...,lmax 
        m = 0,...,l 
        r: real part 
        i: imaginary part
        
        with lmax=3 this arrangement looks like
        
            0 1 2 3       0 1 2 3        r i i i
        l = 1 1 2 3   m = 1 0 1 2    Y = r r i i
            2 2 2 3       2 1 0 1        r r r i
            3 3 3 3       3 2 1 0        r r r r 
        
        the full harmonic with l, m (m>0): Y[l,l-m] + 1.0j*Y[l-m,l] 
                                    (m=0): Y[l,l]
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi = self.cart_to_sph( x, y, z )
        # alp
        Y = np.empty( shape = (self.lmax_p,self.lmax_p,*sin_theta.shape), 
                                dtype = sin_theta.dtype )
        Y[0,0] = np.full_like( sin_theta, self.Yoo )
        Y[1,1] = self.alp_cl[1] * cos_theta * Y[0,0]
        Y[1,0] = self.alp_dl[1] * sin_theta * Y[0,0]
        Y[0,1] = Y[1,0]
        for l in range(2,self.lmax_p):
            Y[l,2:l+1] = self.alp_al[l] * ( cos_theta * Y[l-1,1:l]
                                        + self.alp_bl[l] * Y[l-2,:l-1] )
            Y[l,1] = self.alp_cl[l] * cos_theta * Y[l-1,0]
            Y[l,0] = self.alp_dl[l] * sin_theta * Y[l-1,0]
            Y[:l,l] = Y[l,:l]
        # ylm
        c = cos_phi
        s = sin_phi
        Y[ self.I[1], self.J[1] ] *=  c
        Y[ self.J[1], self.I[1] ] *=  s
        for m in range(2,self.lmax_p):
            c, s = cos_phi * c - sin_phi * s, sin_phi * c + cos_phi * s
            Y[ self.I[m], self.J[m] ] *= c
            Y[ self.J[m], self.I[m] ] *= s
        return r, sin_theta, cos_theta, sin_phi, cos_phi, Y

    
    def ylm_rl( self, x, y, z ):
        """ 
        Returns: r, sin_theta, cos_theta, sin_phi, cos_phi, Y
        Y: r**l * Y_l^m( \theta, \phi )
        ---------------------------------------------------------
        All same as sph_repr.ylm, only with a r^l multiplied 
        to spherical harmonics.
        
        r**l * Y_l^m becomes  (m>0): Y[l,l-m] + 1.0j*Y[l-m,l] 
                              (m=0): Y[l,l]
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi = self.cart_to_sph( x, y, z )
        # r^l preparation
        r_sin_theta = r * sin_theta
        r_cos_theta = r * cos_theta
        r2 = r * r
        # alp
        Y = np.empty( shape = (self.lmax_p,self.lmax_p,*sin_theta.shape), 
                                dtype = sin_theta.dtype )
        Y[0,0] = np.full_like( r_sin_theta, self.Yoo )
        Y[1,1] = self.alp_cl[1] * r_cos_theta * Y[0,0]
        Y[1,0] = self.alp_dl[1] * r_sin_theta * Y[0,0]
        Y[0,1] = Y[1,0]
        for l in range(2,self.lmax_p):
            Y[l,2:l+1] = self.alp_al[l] * ( r_cos_theta * Y[l-1,1:l]
                                        + r2 * self.alp_bl[l] * Y[l-2,:l-1] )
            Y[l,1] = self.alp_cl[l] * r_cos_theta * Y[l-1,0]
            Y[l,0] = self.alp_dl[l] * r_sin_theta * Y[l-1,0]
            Y[:l,l] = Y[l,:l]
        # ylm
        c = cos_phi
        s = sin_phi
        Y[ self.I[1], self.J[1] ] *=  c
        Y[ self.J[1], self.I[1] ] *=  s
        for m in range(2,self.lmax_p):
            c, s = cos_phi * c - sin_phi * s, sin_phi * c + cos_phi * s
            Y[ self.I[m], self.J[m] ] *= c
            Y[ self.J[m], self.I[m] ] *= s
        return r, sin_theta, cos_theta, sin_phi, cos_phi, Y    
    
    
    def ylm_partials( self, sin_theta, cos_theta, Y, with_r=None ):
        """
        Return: Y_theta, Y_phi
        i.e. partial derivatives of spherical harmonics wrt theta, phi
        with_r:
        if r^l is multiplied to spherical harmonics (see sph_repr.ylm_rl), 
        then "with_r = r" is required for the correct evaluation of 
        the partial deriavitives.
        --------------------------------------------------------------
        see ylm for the array storage convention
        """
        # paritial theta
        cot_theta = cos_theta / sin_theta
        Y_theta = cot_theta * self.l * Y
        if with_r is None:
            Y_theta[1:,1:] -= Y[:-1,:-1] * self.coef / sin_theta
        else:
            Y_theta[1:,1:] -= with_r * Y[:-1,:-1] * self.coef / sin_theta
        # partial phi
        axes = list( range(len(Y.shape)) )
        axes[0], axes[1] = 1, 0
        Y_phi = np.transpose( Y, axes=axes ).copy()
        Y_phi[ self.tril_indices ] *= -1
        Y_phi *= self.m
        return Y_theta, Y_phi
    

# test routines ----------------------------------------------------------
def test_sph_repr( n = 1000 ):
    from scipy.special import sph_harm
    lmax = 8
    sph = sph_repr( lmax )
    x = np.random.uniform(-1.0,1.0,size=n)
    y = np.random.uniform(-1.0,1.0,size=n)
    z = np.random.uniform(-1.0,1.0,size=n)
    r, theta, phi = sph.cart_to_angles( x, y, z )
    r, st, ct, sp, cp, Y = sph.ylm( x, y, z )
    r, _,_,_,_, Y_rl = sph.ylm_rl( x, y, z )
    Y_theta, Y_phi = sph.ylm_partials( st, ct, Y )
    Y_theta_rl, Y_phi_rl = sph.ylm_partials( st, ct, Y_rl, with_r=r )
    cott = ct / st
    errors = []
    for l in range(lmax+1):
        # m = 0
        rl = r**l
        tmp = sph_harm( 0, l, phi, theta )
        errors += [ Y[l,l] - tmp, Y_rl[l,l] - rl*tmp ]
        if l>0:
            tmp = np.sqrt(l*(l+1.)) * np.exp(
                -1.0j * phi ) * sph_harm( 1, l, phi, theta )
            errors += [ Y_theta[l,l] - tmp ]
            errors += [ Y_theta_rl[l,l] - rl*tmp ]
        # m > 0
        for m in range(1,l+1):
            tmp = sph_harm( m, l, phi, theta )
            errors += [ Y[l,l-m] + 1.0j*Y[l-m,l] - tmp,
                       Y_rl[l,l-m] + 1.0j*Y_rl[l-m,l] - rl*tmp ]
            # partial wrt phi
            errors += [ Y_phi[l,l-m] + 1.0j*Y_phi[l-m,l] - 1.0j*m*tmp ]
            errors += [ Y_phi_rl[l,l-m] + 1.0j*Y_phi_rl[l-m,l] - 1.0j*rl*m*tmp ]
            # partial wrt theta
            tmp2 = m * cott * tmp 
            if m < l: tmp2 += np.sqrt((l-m)*(l+m+1.)) * np.exp( 
                -1.0j * phi ) * sph_harm( m+1, l, phi, theta )
            errors += [ Y_theta[l,l-m] + 1.0j*Y_theta[l-m,l] - tmp2 ]
            errors += [ Y_theta_rl[l,l-m] + 1.0j*Y_theta_rl[l-m,l] - rl*tmp2 ]

    errors = abs( np.array(errors).reshape(-1) )
    print( """
    comparison with scipy.sph_harm: 
    tests included: ylm, ylm_rl, ylm_partials (with_r= None and r)
    all diffs close to zero: {} 
    max difference: {}
    """.format( np.allclose(errors,0.0), errors.max() ) )


if __name__=='__main__':

    test_sph_repr()

