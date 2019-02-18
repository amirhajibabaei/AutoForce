
# coding: utf-8

# In[ ]:


import numpy as np
from theforce.sph_repr import sph_repr
from math import factorial as fac
from theforce.sphcart import sph_vec_to_cart


#                           --- radial functions ---
class gaussian:
    
    def __init__( self, sigma ):
        """ exp( - r^2 / 2 sigma^2 ) """
        self.alpha = -1. / (sigma*sigma)
    
    def radial( self, r ):
        x = self.alpha * r
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

    
class poly_cutoff:
    
    def __init__( self, rc, n ):
        """ ( 1 - r / r_c )^n """
        self.rc = rc
        self.n  = n
        self.n_ = n - 1 
    
    def radial( self, r ):
        x = 1. - r/self.rc
        x[ np.where(x<0.0) ] = 0.0
        y = x**(self.n_)
        return x*y, -self.n*y/self.rc

    
    
#                              --- soap descriptor ---
class sesoap:
    
    def __init__( self, lmax, nmax, radial ):
        """
        lmax: maximum l in r^l * Ylm terms (Ylm are spherical harmonics)
        nmax: maximum n in r^(2n) terms
        radial: radial function e.g. gaussian (original) or quadratic_cutoff
        """
        self.lmax = lmax
        self.nmax = nmax
        self.radial = radial
        
        self.sph = sph_repr( lmax )
        
        # prepare some stuff
        self._m = [ ( [ l for m in range(0,l+1) ] + [ m for m in range(0,l) ],
                  [ m for m in range(0,l+1) ] + [ l for m in range(0,l) ] )  
                   for l in range(0,lmax+1) ]
        
        # lower triangle indices
        self.In, self.Jn = np.tril_indices(nmax+1)

        # l,n,n'-dependent constant 
        a_ln = np.array( [ [ 1. / ( (2*l+1) * 2**(2*n+l) * fac(n) * fac(n+l) ) 
                            for n in range(nmax+1) ] for l in range(lmax+1) ] )
        tmp = self.compress( np.sqrt(np.einsum('ln,lm->lnm',a_ln,a_ln) ), 'lnn' )
        self.lnnp_c = [ tmp, tmp[:,np.newaxis], tmp[:,np.newaxis,np.newaxis] ]
        
        # prepare for broadcasting
        self.rns = 2 * np.arange(self.nmax+1).reshape(nmax+1,1,1,1)

        
        
    def descriptor( self, x, y, z ):
        """
        Inputs:   x,y,z -> Cartesian coordinates
        Returns:  p -> compressed (1d) descriptor
        """
        r, _,_,_,_, Y = self.sph.ylm_rl(x,y,z)
        R, _ = self.radial.radial( r )
        s = ( R * r**self.rns * Y ).sum(axis=-1)
        return self.soap_dot(s,s,reverse=False)
    
    
    def derivatives( self, x, y, z, sumj=True, cart=True ):
        """
        Inputs:   x,y,z -> Cartesian coordinates
        Returns:  p, q
        sumj:     perform the summation over atoms
        ---------------------------------------
        p:        compressed (1d) descriptor
        q:        [dp_dx, dp_dy, dp_dz] 
                  (or gradient in sph coords if cart=False)
        """
        r, sin_theta, cos_theta, sin_phi, cos_phi, Y = self.sph.ylm_rl(x,y,z)
        Y_theta, Y_phi = self.sph.ylm_partials( sin_theta, cos_theta, Y, with_r=r )
        R, dR = self.radial.radial( r ) 
        rns = r**self.rns
        R_rns = R * rns
        # descriptor
        s = ( R_rns * Y ).sum(axis=-1)
        p = self.soap_dot(s,s,reverse=False)
        # gradients
        rns_plus_l = self.rns + self.sph.l
        Y_theta   /= r
        Y_phi     /= r*sin_theta
        q1 = self.soap_dot( s, ((dR * rns + R_rns * rns_plus_l / r) * Y), jb='j'  )
        q2 = self.soap_dot( s, (R_rns * Y_theta), jb='j'  )
        q3 = self.soap_dot( s, (R_rns * Y_phi), jb='j'  )
        if cart: 
            q1, q2, q3 = sph_vec_to_cart( sin_theta, cos_theta, sin_phi, cos_phi, q1,q2,q3 )
        if sumj:
            return p, np.array([q1, q2, q3]).sum(axis=-1)            
        else:
            return p, np.array([q1, q2, q3]) 

    
    # ------------------- convenience functions -------------------------------------------------

    def soap_dot( self, a, b, ja='', jb='', reverse=True ):
        jc = ja
        if   ja==jb: 
            jd = ja; jr = ja
        elif ja!=jb: 
            jd = ja+jb; jr = jb+ja
        c = np.einsum('n...'+ja+',m...'+jb+'->nm...'+jd,a,b)
        if reverse: c += np.einsum('n...'+jb+',m...'+ja+'->nm...'+jr,b,a)
        return self.compress( self.sum_all_m(c), 'lnn'+jd ) * self.lnnp_c[len(jd)]
        
    
    def sum_all_m( self, c ):
        rank = len(c.shape)
        if rank==6:
            return np.array( [ 2*c[:,:,self._m[l][0],self._m[l][1],:,:].sum(axis=-3) 
                    - c[:,:,l,l,:,:] for l in range(self.lmax+1) ] )
        elif rank==5:
            return np.array( [ 2*c[:,:,self._m[l][0],self._m[l][1],:].sum(axis=-2) 
                    - c[:,:,l,l,:] for l in range(self.lmax+1) ] )
        elif rank==4:
            return np.array( [ 2*c[:,:,self._m[l][0],self._m[l][1]].sum(axis=-1) 
                    - c[:,:,l,l] for l in range(self.lmax+1) ] )
    
    
    def compress( self, a, type ):
        if type=='lnn': 
            return a[:,self.In,self.Jn].reshape(-1)
        elif type=='lnnj' or type=='lnnk': 
            j = a.shape[-1]
            return a[:,self.In,self.Jn].reshape(-1,j)
        elif type=='3lnn':
            return a[:,:,self.In,self.Jn].reshape((3,-1))
        elif type=='3lnnj' or type=='3lnnk':
            j = a.shape[-1]
            return a[:,:,self.In,self.Jn,:].reshape((3,-1,j))
        elif type=='lnnjk' or type=='lnnkj': 
            j = a.shape[-1]
            return a[:,self.In,self.Jn].reshape(-1,j,j)
        else:
            print("type {} not defined yet, for matrix with shape {}".format(type,a.shape))
        
    
    def decompress( self, v, type, n=None, l=None ):
        if n is None: n = self.nmax
        if l is None: l = self.lmax
        d = len( v.shape )
        if type=='lnn':
            a = np.empty(shape=(l+1,n+1,n+1),dtype=v.dtype)
            a[:,self.In,self.Jn] = v.reshape((l+1, (n+1)*(n+2)//2))
            a[:,self.Jn,self.In] = a[:,self.In,self.Jn]
        elif type=='lnnj' or type=='lnnk':
            j = v.shape[-1]
            a = np.empty(shape=(l+1,n+1,n+1,j),dtype=v.dtype)
            a[:,self.In,self.Jn,:] = v.reshape((l+1, (n+1)*(n+2)//2,j))
            a[:,self.Jn,self.In,:] = a[:,self.In,self.Jn,:]
        elif type=='3lnn':
            a = np.empty(shape=(3,l+1,n+1,n+1),dtype=v.dtype)
            a[:,:,self.In,self.Jn] = v.reshape((3,l+1, (n+1)*(n+2)//2))
            a[:,:,self.Jn,self.In] = a[:,:,self.In,self.Jn]
        elif type=='3lnnj' or type=='3lnnk':
            j = v.shape[-1]
            a = np.empty(shape=(3,l+1,n+1,n+1,j),dtype=v.dtype)
            a[:,:,self.In,self.Jn,:] = v.reshape((3,l+1, (n+1)*(n+2)//2,j))
            a[:,:,self.Jn,self.In,:] = a[:,:,self.In,self.Jn,:]
        elif type=='lnnjk' or type=='lnnkj':
            j = v.shape[-1]
            a = np.empty(shape=(l+1,n+1,n+1,j,j),dtype=v.dtype)
            a[:,self.In,self.Jn,:] = v.reshape((l+1, (n+1)*(n+2)//2,j,j))
            a[:,self.Jn,self.In,:] = a[:,self.In,self.Jn,:,:]
        else:
            print("type {} not defined yet, for matrix with shape {}".format(type,a.shape))
        return a

    
    
# tests ----------------------------------------------------------------------------------
    
def test_sesoap():
    """ trying to regenerate numbers obtained by symbolic calculations using sympy """
    from theforce.sphcart import cart_vec_to_sph, rotate
    x = np.array( [0.175, 0.884, -0.87, 0.354, -0.082] )
    y = np.array( [-0.791, 0.116, 0.19, -0.832, 0.184] )
    z = np.array( [0.387, 0.761, 0.655, -0.528, 0.973] )
    env = sesoap( 2, 2, quadratic_cutoff(3.0) )
    p_ = env.descriptor( x, y, z )
    p_dc, q_dc = env.derivatives( x, y, z, cart=False )
    p_ = env.decompress( p_, 'lnn' )
    p_d = env.decompress( p_dc, 'lnn' )
    q_d = env.decompress( q_dc, '3lnn' )
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
    ref_p *= env.decompress( env.lnnp_c[0], 'lnn' )


    print( "\nTesting validity of sesoap ...")
    print( np.allclose( p_-ref_p[0], 0.0 ) ) 
    print( np.allclose( p_d-ref_p[0], 0.0 ) ) 
    for k in range(3):
        print( np.allclose( q_d[k]-ref_p[k+1], 0.0 ) )
        
    
    pj, qj = env.derivatives(x,y,z,sumj=False, cart=True)
    pj_ = env.decompress( pj, 'lnn' )
    qj_ = env.decompress( qj, '3lnnj' )
    h = np.array( cart_vec_to_sph(x,y,z,qj_[0],qj_[1],qj_[2]) )
    print( np.allclose( h.sum(axis=-1)-ref_p[1:], 0.0 ) )
    
    
    pj, qj = env.derivatives(x,y,z,sumj=True, cart=True)
    pj_ = env.decompress( pj, 'lnn' )
    qj__ = env.decompress( qj, '3lnn' )
    print( np.allclose( qj_.sum(axis=-1)-qj__, 0.0 ) )
        
    # test rotations
    axis = np.random.uniform(size=3)
    beta = np.random.uniform(0,2*np.pi)
    xx,yy,zz = rotate(x,y,z,axis,beta)
    p, q = env.derivatives( x, y, z, sumj=False, cart=True )
    pp, qq = env.derivatives( xx, yy, zz, sumj=False, cart=True )
    rot = np.array( rotate(q[0],q[1],q[2],axis,beta) )
    print( np.allclose(rot-qq,0.0) )
        
        
def test_sesoap_performance( n=30, N=100 ):
    import time
    print("\nTesting speed of sesoap with random xyz[{},3]".format(n))
    
    # np.random
    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
    finish = time.time()
    delta1 = (finish-start)/N
    print( "t1: {} Sec per np.random.uniform(shape=({},3))".format( delta1, n ) )
    
    env = sesoap( 5, 5, quadratic_cutoff(3.0) )

    # descriptor
    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
        p = env.descriptor( x, y, z )
    finish = time.time()
    delta2 = (finish-start)/N
    print( "t2: {} Sec per descriptor".format( delta2 ) )
    # derivatives
    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
        p, q = env.derivatives( x, y, z )
    finish = time.time()
    delta3 = (finish-start)/N
    print( "t3: {} Sec per derivatives (j-reduced)".format( delta3 ) )

    start = time.time()
    for _ in range(N):
        x, y, z = ( np.random.uniform(-1.,1.,size=n) for _ in range(3) )
        p, q = env.derivatives( x, y, z, sumj=False, cart=True )
    finish = time.time()
    delta4 = (finish-start)/N
    print( "t4: {} Sec per full derivatives".format( delta4 ) )
    
    print( "performance measure t2/t1: {}\n".format(delta2/delta1) )
        
        
if __name__=='__main__':
    
    test_sesoap()
    
    test_sesoap_performance()
    
    

