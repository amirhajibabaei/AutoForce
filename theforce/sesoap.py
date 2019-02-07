
# coding: utf-8

# In[ ]:


import numpy as np
from theforce.sph_repr import sph_repr
from math import factorial as fac



#                           --- radial functions ---
class exponential:
    
    def __init__( self, alpha ):
        """ exp( -alpha r^2 / 2 ) """
        self.alpha = alpha
    
    def radial( self, r ):
        x = -self.alpha * r
        y = np.exp(x*r/2)
        return y, x*y


class quadratic_cutoff:
    
    def __init__( self, rc ):
        """ ( 1 - r / r_c )^2 """
        self.rc = rc
    
    def radial( self, r ):
        x = 1. - r/self.rc
        x[ np.where(x<0.0) ] = 0.0
        return x*x, -2*x/self.rc
    


    
    
#                              --- bare soap ---
class sesoap:
    
    def __init__( self, lmax, nmax, radial ):
        """
        lmax: maximum l index in spherical harmonics Ylm
        nmax: maximum n power in r^(2n) 
        radial: e.g. exponential or quadratic_cutoff
        """
        self.lmax = lmax
        self.nmax = nmax
        self.radial = radial
        
        self.sph = sph_repr( lmax )
        
        # prepare some stuff
        self.L = self.sph.l[:,:,0]
        self._m = [ ( [ l for m in range(0,l+1) ] + [ m for m in range(0,l) ],
                  [ m for m in range(0,l+1) ] + [ l for m in range(0,l) ] )  
                   for l in range(0,lmax+1) ]
        
        # l,n,n'-dependent constant 
        a_ln = np.array( [ [ 1. / ( (2*l+1) * 2**(2*n+l) * fac(n) * fac(n+l) ) 
                            for n in range(nmax+1) ] for l in range(lmax+1) ] )
        self.lnnp_cost = np.sqrt( np.einsum('ln,lm->lnm',a_ln,a_ln) )
        
        
        
        
    def get_alpha_as_tensor( self, alpha ):
        """
        returns alpha^(l+n+n') as an array 
        with shape (lmax+1,nmax+1,nmax+1) 
        """
        n = max(self.nmax,self.lmax)
        t = [ alpha**k for k in range(n+1) ]
        return np.einsum('i,j,k->ijk',t[:self.lmax+1],
                         t[:self.nmax+1],t[:self.nmax+1])
    
    
    def create_alpha_tensor( self, alpha ):
        """
        creates an "alpha" attribute which contains
        alpha^(l+n+n') as an array with shape 
        (lmax+1,nmax+1,nmax+1)
        """
        self.alpha = self.get_alpha_as_tensor( alpha )
        
        

    def soap0( self, x, y, z ):
        """
        Inputs: x,y,z -> Cartesian coordinates
        Returns: p -> array with shape (lmax+1,nmax+1,nmax+1)
        """
        r, _,_,_,_, Y = self.sph.ylm_rl(x,y,z)
        R, _ = self.radial.radial( r )
        k  = R * Y
        K  = [ k.sum(axis=-1) ]
        r2 = r*r
        for n in range(1,self.nmax+1):
            k *= r2
            K += [ k.sum(axis=-1) ]
        K = np.einsum( 'ilm,jlm->ijlm', K, K )
        p = np.array( [ 2*K[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
             - K[:,:,l,l] for l in range(self.lmax+1) ] )
        p *= self.lnnp_cost
        return p 
    
    
    
    def soap0del( self, x, y, z ):
        """
        Inputs: x,y,z -> Cartesian coordinates
        Returns: p, p_r, p_theta, p_phi -> 
                arrays with shape (lmax+1,nmax+1,nmax+1)
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi, Y = self.sph.ylm_rl(x,y,z)
        Y_theta, Y_phi = self.sph.ylm_partials( sin_theta, cos_theta, Y, with_r=r )
        R, dR = self.radial.radial( r )
        # n=0 terms
        k     = R * Y
        R1    = dR * Y 
        R2    = ( R / r ) * Y
        Theta = R * Y_theta / r
        Phi   = R * Y_phi / (r*sin_theta)
        # sum over j
        scalar     = [ k.sum(axis=-1) ]
        grad_r     = [ R1.sum(axis=-1) + R2.sum(axis=-1) * self.L ]
        grad_theta = [ Theta.sum(axis=-1) ]
        grad_phi   = [ Phi.sum(axis=-1) ]
        # r^2n multipliers
        r2 = r*r
        for n in range(1,self.nmax+1):
            k     *= r2
            R1    *= r2
            R2    *= r2
            Theta *= r2
            Phi   *= r2
            # ---
            scalar     += [ k.sum(axis=-1) ]
            grad_r     += [ R1.sum(axis=-1) + R2.sum(axis=-1) * (2*n+self.L) ]
            grad_theta += [ Theta.sum(axis=-1) ]
            grad_phi   += [ Phi.sum(axis=-1) ]
        # n, n' coupling 
        c       = np.einsum( 'nlm,klm->nklm', scalar, scalar )
        c_r     = np.einsum( 'nlm,klm->nklm', scalar, grad_r ) + \
                  np.einsum( 'nlm,klm->nklm', grad_r, scalar )
        c_theta = np.einsum( 'nlm,klm->nklm', scalar, grad_theta ) + \
                  np.einsum( 'nlm,klm->nklm', grad_theta, scalar )
        c_phi   = np.einsum( 'nlm,klm->nklm', scalar, grad_phi ) + \
                  np.einsum( 'nlm,klm->nklm', grad_phi, scalar )
        # sum over m
        p       = np.array( [ 2*c[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
                 - c[:,:,l,l] for l in range(self.lmax+1) ] )
        p_r     = np.array( [ 2*c_r[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
                 - c_r[:,:,l,l] for l in range(self.lmax+1) ] )
        p_theta = np.array( [ 2*c_theta[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
                 - c_theta[:,:,l,l] for l in range(self.lmax+1) ] )
        p_phi   = np.array( [ 2*c_phi[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
                 - c_phi[:,:,l,l] for l in range(self.lmax+1) ] )
        # apply l,n,n'-dependent coef
        p *= self.lnnp_cost
        q = np.array([p_r, p_theta, p_phi]) * self.lnnp_cost
        return p, q
    
    
    
    
    
# tests ------------------------------------------------------------------------------
    
def test_sesoap():
    """ trying to regenerate numbers obtained by symbolic calculations using sympy """
    x = np.array( [0.175, 0.884, -0.87, 0.354, -0.082] )
    y = np.array( [-0.791, 0.116, 0.19, -0.832, 0.184] )
    z = np.array( [0.387, 0.761, 0.655, -0.528, 0.973] )
    env = sesoap( 2, 2, quadratic_cutoff(3.0) )
    p_ = env.soap0( x, y, z )
    p_d, q_d = env.soap0del( x, y, z )
    ref_p = [          np.array([[[0.36174603, 0.39013356, 0.43448023],
                                 [0.39013356, 0.42074877, 0.46857549],
                                 [0.43448023, 0.46857549, 0.5218387 ]],

                                [[0.2906253 , 0.30558356, 0.33600938],
                                 [0.30558356, 0.3246583 , 0.36077952],
                                 [0.33600938, 0.36077952, 0.40524778]],

                                [[0.16241845, 0.18307552, 0.20443194],
                                 [0.18307552, 0.22340802, 0.26811937],
                                 [0.20443194, 0.26811937, 0.34109511]]]),
                       np.array([[[-0.73777549, -0.05089412,  0.74691856],
                                 [-0.05089412,  0.74833475,  1.70005743],
                                 [ 0.74691856,  1.70005743,  2.85847646]],

                                [[-0.01237519,  0.56690766,  1.23261539],
                                 [ 0.56690766,  1.21157686,  1.99318763],
                                 [ 1.23261539,  1.99318763,  2.95749108]],

                                [[ 0.27361894,  0.63696076,  1.08095971],
                                 [ 0.63696076,  1.15336381,  1.84451275],
                                 [ 1.08095971,  1.84451275,  2.9120592 ]]]),
                       np.array([[[ 0.        ,  0.        ,  0.        ],
                                 [ 0.        ,  0.        ,  0.        ],
                                 [ 0.        ,  0.        ,  0.        ]],

                                [[-0.81797727, -0.88483089, -0.99106192],
                                 [-0.88483089, -0.95446211, -1.06668809],
                                 [-0.99106192, -1.06668809, -1.18983543]],

                                [[ 0.03152424,  0.0597677 ,  0.07161054],
                                 [ 0.0597677 ,  0.11466049,  0.15943685],
                                 [ 0.07161054,  0.15943685,  0.24410156]]]),
                       np.array([[[ 0.        ,  0.        ,  0.        ],
                                 [ 0.        ,  0.        ,  0.        ],
                                 [ 0.        ,  0.        ,  0.        ]],

                                [[ 0.01059708,  0.00517264, -0.00218289],
                                 [ 0.00517264, -0.00037216, -0.00786604],
                                 [-0.00218289, -0.00786604, -0.01549284]],

                                [[ 0.02103876,  0.00576316, -0.01632531],
                                 [ 0.00576316, -0.01022614, -0.03301236],
                                 [-0.01632531, -0.03301236, -0.0564123 ]]])]
    ref_p *= env.lnnp_cost

    print( "\nTesting sesoap ...")
    print( np.allclose( p_-ref_p[0], 0.0 ) ) 
    print( np.allclose( p_d-ref_p[0], 0.0 ) ) 
    for k in range(3):
        print( np.allclose( q_d[k]-ref_p[k+1], 0.0 ) )

        
        
def test_sesoap_performance( n=30, N=100 ):
    import time
    print("\nTesting speed of sesoap ...")
    
    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
    finish = time.time()
    delta1 = (finish-start)/N
    print( "t1: {} Sec per random xyz[{},3]".format( delta1, n ) )
    
    
    env = sesoap( 6, 6, quadratic_cutoff(3.0) )
    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
        p, q = env.soap0del( x, y, z )
    finish = time.time()
    delta2 = (finish-start)/N
    print( "t2: {} Sec per soap of xyz[{},3]".format( delta2, n ) )
    
    print( "performance measure t2/t1: {}\n".format(delta2/delta1) )
        
        
if __name__=='__main__':
    
    test_sesoap()
    
    test_sesoap_performance()
    
    

