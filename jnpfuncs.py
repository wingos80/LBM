import jax.numpy as jnp
from settings import *

## Initial conditions
def flow_taylorgreen(t=0, tau=1, rho0=1, u_max=0.2)-> jnp.ndarray:
    nu=(2*tau-1)/6
    x = jnp.arange(XMAX)+0.5
    y = jnp.arange(YMAX)+0.5
    [X, Y] = jnp.meshgrid(x,y)
    kx = 2*jnp.pi/XMAX
    ky = 2*jnp.pi/YMAX
    td = 1/(nu*(kx*kx+ky*ky))
    
    u0 = jnp.array([-u_max*jnp.sqrt(ky/kx)*jnp.cos(kx*X)*jnp.sin(ky*Y)*jnp.exp(-t/td),
                   u_max*jnp.sqrt(kx/ky)*jnp.sin(kx*X)*jnp.cos(ky*Y)*jnp.exp(-t/td)])
    P = -0.25*rho0*u_max*u_max*((ky/kx)*jnp.cos(2*kx*X)+(kx/ky)*jnp.cos(2*ky*Y))*jnp.exp(-2*t/td)
    rho = rho0+3*P
    u = jnp.zeros((XMAX,YMAX,2))
    u[:,:,0] = u0[0,:,:].T
    u[:,:,1] = u0[1,:,:].T

    f = calculate_f_eq(rho.T, u)
    return f


def flow_up()-> jnp.ndarray:
    rho = jnp.ones((XMAX,YMAX))
    u = jnp.ones((XMAX,YMAX,2))
    u[:,:,0] = 0  # make ux 0 
    u[:,:-3,1] = 0  # make ux 0 
    u[:,-2:,1] = 0  # make ux 0 
    f = calculate_f_eq(rho, u)
    return f


def flow_right()-> jnp.ndarray:
    rho = jnp.ones((XMAX,YMAX))
    # rho[-3,:] += 1
    # u = 0.15*jnp.ones((XMAX,YMAX,2))
    u = jnp.stack((0.15*jnp.ones((XMAX,YMAX,1)),jnp.zeros((XMAX,YMAX,1))),axis=2)
    # u.at[:,:,1].set(0)  # make uy 0
    # u[:-3,:,0] = 0  # make ux 0 
    # u[-2:,:,0] = 0  # make ux 0 
    f = calculate_f_eq(rho, u)
    return f


def flow_random()-> jnp.ndarray:
    spread = 0.1
    rho = jnp.ones((XMAX,YMAX))
    u = jnp.ones((XMAX,YMAX,2))
    u[:,:,0] = 0  # make ux 0 
    u[:,:-3,1] = 0  # make ux 0 
    u[:,-2:,1] = 0  # make ux 0 
    u += spread*(jnp.random.rand(XMAX,YMAX,2)-0.5)
    f = calculate_f_eq(rho, u)
    return f

## Solid material creation
def create_solids()-> jnp.ndarray:
    solids = jnp.logical_or(Y == 0, Y==YMAX-1)  # solid top and bottom walls
    circle_center = XMAX/4, YMAX/2+4
    circle_radius = YMAX/10
    solids += (X - circle_center[0])**2 + (Y - circle_center[1])**2 < (circle_radius)**2
    return solids


## Moments of f
def calculate_rho(f: jnp.ndarray)-> jnp.ndarray:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :returns: density, scalar field, particles per cell
    """
    return jnp.sum(f, axis=2)


def calculate_momentum(f: jnp.ndarray,c=None)-> jnp.ndarray:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :param c: basis velocity components
    :returns: momentum, vector field, momentum per cell
    """
    if c is not None:
        ## Generic implementation
        cx, cy = c[0], c[1]
        momentum_x = jnp.sum(f*cx,axis=2)
        momentum_y = jnp.sum(f*cy,axis=2)
    else:
        ## D2Q9 implementation
        momentum_x = (f[:,:,1]+f[:,:,5]+f[:,:,8]) - (f[:,:,3]+f[:,:,6]+f[:,:,7])
        momentum_y = (f[:,:,2]+f[:,:,5]+f[:,:,6]) - (f[:,:,4]+f[:,:,7]+f[:,:,8])
    return jnp.stack((momentum_x, momentum_y), axis=2)


def calculate_velocity(momentum: jnp.ndarray, rho: jnp.ndarray)-> jnp.ndarray:
    """
    :param momentum: momentum, vector field, how many particles moving in each direction per cell
    :param rho: density, scalar field, particles per cell
    :return: velocity
    """
    ux = momentum[:,:,0]/rho
    uy = momentum[:,:,1]/rho
    u = jnp.stack((ux, uy), axis=2)
    return u


def calculate_f_eq(rho: jnp.ndarray, u: jnp.ndarray)-> jnp.ndarray:
    """
    Returns the equillibrium population/particle field. 
    Implementation is specific for D2Q9, meaning that the weights of 
    D2Q9's velocity set are used and the generic equation for f_eq is 
    expanded using those weights, resulting in the hard coded coefficients
    of rho. 
    TODO, add generic implementation
    :param rho: density, scalar field, particles per cell
    :param u: velocity, vector field, how many particles moving in each direction per cell
    :return: equillibrium population
    """
    ## D2Q9 implementation
    ux, uy = u[:,:,0].squeeze(), u[:,:,1].squeeze()

    # First pre compute commonly used terms
    rho_1 = 2*rho/9
    rho_2 = rho/18
    rho_3 = rho/36

    u2 = ux**2 + uy**2
    ux2 = ux**2
    uy2 = uy**2
    uxuy = ux*uy

    # Use the pre-computed terms to compute equillibrium distributions
    f_eq_0 = rho_1*(2 - 3*u2)
    f_eq_1 = rho_2*(2 + 6*ux + 9*ux2 - 3*u2)
    f_eq_2 = rho_2*(2 + 6*uy + 9*uy2 - 3*u2)
    f_eq_3 = rho_2*(2 - 6*ux + 9*ux2 - 3*u2)
    f_eq_4 = rho_2*(2 - 6*uy + 9*uy2 - 3*u2)
    f_eq_5 = rho_3*(1 + 3*(ux + uy) + 9*uxuy +3*u2)
    f_eq_6 = rho_3*(1 - 3*(ux - uy) - 9*uxuy +3*u2)
    f_eq_7 = rho_3*(1 - 3*(ux + uy) + 9*uxuy +3*u2)
    f_eq_8 = rho_3*(1 + 3*(ux - uy) - 9*uxuy +3*u2)
    f_eq = jnp.stack((f_eq_0,f_eq_1,f_eq_2,f_eq_3,f_eq_4,f_eq_5,f_eq_6,f_eq_7,f_eq_8),axis=2)

    return f_eq


## Simulation steps
def collision_step(f: jnp.ndarray, f_eq: jnp.ndarray, dt=1, tau=1)-> jnp.ndarray:
    """
    BGK collision step
    :param f: array containing the current population
    :param f_eq: array containing the equillibrium population
    :retunr: population after collision
    """
    return f*(1-dt/tau) + dt/tau*f_eq


def streaming_step(f: jnp.ndarray)-> jnp.ndarray:
    # Streaming
    f_streamed = jnp.stack([jnp.roll(jnp.roll(f[:,:,i], 
                                              c[0,i], 
                                              axis=0), 
                                      c[1,i], 
                                      axis=1) 
                            for i in range(9)], 
                            axis=2)
    return f_streamed


def BC_solids(f: jnp.ndarray, solids: jnp.ndarray)-> jnp.ndarray:
    b = f[solids,:]
    b = b[:,[0,3,4,1,2,7,8,5,6]]
    f = f.at[solids,:].set(b)
    return f

