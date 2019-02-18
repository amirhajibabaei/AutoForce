
# coding: utf-8

# In[ ]:


import numpy as np


# ---------------------------------------------------- conversions
def cart_to_angles( x, y, z ):
    rxy_sq = x*x + y*y
    r      = np.sqrt( rxy_sq + z*z )
    theta  = np.arctan2( np.sqrt(rxy_sq), z )
    phi    = np.arctan2( y, x )
    return r, theta, phi


def angles_to_cart( r, theta, phi ):
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return x, y ,z


def cart_to_sph( x, y, z, tiny=1.0e-16 ):
    rxy_sq    = x*x + y*y 
    rxy       = np.sqrt( rxy_sq ) + tiny
    r_sq      = rxy_sq + z*z
    r         = np.sqrt( r_sq )
    sin_theta = rxy/r
    cos_theta = z/r
    sin_phi   = y/rxy
    cos_phi   = x/rxy
    return r, sin_theta, cos_theta, sin_phi, cos_phi


def sph_to_cart( sin_theta, cos_theta, sin_phi, cos_phi, F_r, F_theta, F_phi ):
    F_x = sin_theta * cos_phi * F_r + cos_theta * cos_phi * F_theta - sin_phi * F_phi
    F_y = sin_theta * sin_phi * F_r + cos_theta * sin_phi * F_theta + cos_phi * F_phi
    F_z = cos_theta * F_r - sin_theta * F_theta
    return F_x, F_y, F_z


def angles_to_sph( r, theta, phi ):
    return r, np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)



# ------------------------------------------------------- rotations
def rotation( axis, theta ):
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def euler_rotation( alpha, beta, gamma ):
    yhat = np.asarray([0,1.,0])
    zhat = np.asarray([0,0,1.])
    R1 = rotation( zhat, gamma )
    R2 = rotation( yhat, beta  )
    R3 = rotation( zhat, alpha )
    return np.dot( R3, np.dot( R2, R1 ) )


def rotate( a, b, c, axis, beta, angles=False ):
    """ a,b,c =  x,y,z   or   r, theta, phi """
    if angles:
        x,y,z = angles_to_cart(a,b,c)
    else:
        x,y,z = a,b,c
    rmat = rotation( axis, beta )
    # c = np.matmul( rmat, np.asarray([x,y,z]) )
    c =[ rmat[0,0]*x+rmat[0,1]*y+rmat[0,2]*z,
         rmat[1,0]*x+rmat[1,1]*y+rmat[1,2]*z,
         rmat[2,0]*x+rmat[2,1]*y+rmat[2,2]*z ]
    if angles:
        return cart_to_angles( c[0],c[1],c[2] )
    else:
        return c[0],c[1],c[2]




# ----------------------------------------------------------- tests
def rand_cart( N, d=1.0 ):
    x, y, z = ( np.random.uniform(-d,d,size=N) for _ in range(3) )
    return x, y, z

def rand_angles( N, R=1.0 ):
    r = np.random.uniform(0,R,size=N)
    t = np.random.uniform(0,np.pi,size=N)
    p = np.random.uniform(0,2*np.pi,size=N)
    return r, t, p



def test_transforms( N=1000 ):
    r, t, p = rand_angles( N )
    x, y, z = angles_to_cart( r, t, p )
    r2, t2, p2 = cart_to_angles( x, y, z )
    p2 = p2%(2*np.pi)
    test1 =  np.allclose( [r-r2, t-t2, p-p2], 0.0 ) 
    x, y, z = rand_cart( N )
    r, t, p = cart_to_angles( x, y, z )
    x2, y2, z2 = angles_to_cart( r, t, p )
    test2 = np.allclose( [x-x2, y-y2, z-z2], 0.0)
    if test1 and test2:
        print( 'trans test passed' )
    else:
        print( 'trans test failed' )

        
        
def test_rotate( N=1000 ):
    for _ in range(N):
        axis = np.random.uniform(size=3)
        theta1 = np.random.uniform(0,2*np.pi)
        theta2 = np.random.uniform(0,2*np.pi)
        theta = theta1 + theta2
        r, t, p = rand_angles(17,R=10.)
        r1, t1, p1 = rotate( r, t, p, axis, theta1, angles=True )
        r2, t2, p2 = rotate( r1, t1, p1, axis, theta2, angles=True  )
        r3, t3, p3  = rotate( r, t, p, axis, theta, angles=True  )
        test = np.allclose( [r3-r2,t3-t2,p3-p2], 0.0)
        if not test: break
    if test:
        print( 'rot test passed' )
    else:
        print( 'rot test failed' )



if __name__=='__main__':
    
    test_transforms()
    
    test_rotate()

