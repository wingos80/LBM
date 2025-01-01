import numpy as np
from settings import *

## Initial conditions
def flow_taylorgreen(t=0, tau=1, rho0=1, u_max=0.2)-> np.ndarray:
    nu=(2*tau-1)/6
    x = np.arange(XMAX)+0.5
    y = np.arange(YMAX)+0.5
    [X, Y] = np.meshgrid(x,y)
    kx = 2*np.pi/XMAX
    ky = 2*np.pi/YMAX
    td = 1/(nu*(kx*kx+ky*ky))
    
    u0 = np.array([-u_max*np.sqrt(ky/kx)*np.cos(kx*X)*np.sin(ky*Y)*np.exp(-t/td),
                   u_max*np.sqrt(kx/ky)*np.sin(kx*X)*np.cos(ky*Y)*np.exp(-t/td)])
    P = -0.25*rho0*u_max*u_max*((ky/kx)*np.cos(2*kx*X)+(kx/ky)*np.cos(2*ky*Y))*np.exp(-2*t/td)
    rho = rho0+3*P
    u = np.zeros((XMAX,YMAX,2))
    u[:,:,0] = u0[0,:,:].T
    u[:,:,1] = u0[1,:,:].T

    f = calculate_f_eq(rho.T, u)
    return f


def flow_up()-> np.ndarray:
    rho = np.ones((XMAX,YMAX))
    u = np.ones((XMAX,YMAX,2))
    u[:,:,0] = 0  # make ux 0 
    u[:,:-3,1] = 0  # make ux 0 
    u[:,-2:,1] = 0  # make ux 0 
    f = calculate_f_eq(rho, u)
    return f


def flow_right()-> np.ndarray:
    rho = np.ones((XMAX,YMAX))
    # rho[-3,:] += 1
    u = 0.15*np.ones((XMAX,YMAX,2))
    u[:,:,1] = 0  # make uy 0
    # u[:-3,:,0] = 0  # make ux 0 
    # u[-2:,:,0] = 0  # make ux 0 
    f = calculate_f_eq(rho, u)
    return f


def flow_random()-> np.ndarray:
    spread = 0.1
    rho = np.ones((XMAX,YMAX))
    u = np.ones((XMAX,YMAX,2))
    u[:,:,0] = 0  # make ux 0 
    u[:,:-3,1] = 0  # make ux 0 
    u[:,-2:,1] = 0  # make ux 0 
    u += spread*(np.random.rand(XMAX,YMAX,2)-0.5)
    f = calculate_f_eq(rho, u)
    return f

## Solid material creation

def create_solids()-> np.ndarray:
    solids = np.logical_or(Y == 0, Y==YMAX-1)  # solid top and bottom walls
    # solids = np.full(Y.shape, False)
    # solids[:,:] = False
    circle_center = XMAX/4, YMAX/2+4
    circle_radius = YMAX/5
    solids += (X - circle_center[0])**2 + (Y - circle_center[1])**2 < (circle_radius)**2
    return solids


## Moments of f
def calculate_rho(f: np.ndarray)-> np.ndarray:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :returns: density, scalar field, particles per cell
    """
    return np.sum(f, axis=2)


def calculate_momentum(f: np.ndarray,c=None)-> np.ndarray:
    """
    :param f: population, particle vector field, how many particles moving in each direction per cell
    :param c: basis velocity components
    :returns: momentum, vector field, momentum per cell
    """
    if c is not None:
        ## Generic implementation
        cx, cy = c[0], c[1]
        momentum_x = np.sum(f*cx,axis=2)
        momentum_y = np.sum(f*cy,axis=2)
    else:
        ## D2Q9 implementation
        momentum_x = (f[:,:,1]+f[:,:,5]+f[:,:,8]) - (f[:,:,3]+f[:,:,6]+f[:,:,7])
        momentum_y = (f[:,:,2]+f[:,:,5]+f[:,:,6]) - (f[:,:,4]+f[:,:,7]+f[:,:,8])
    return np.stack((momentum_x, momentum_y), axis=2)


def calculate_velocity(momentum: np.ndarray, rho: np.ndarray)-> np.ndarray:
    """
    :param momentum: momentum, vector field, how many particles moving in each direction per cell
    :param rho: density, scalar field, particles per cell
    :return: velocity
    """
    u = np.zeros((XMAX,YMAX,2))
    u[:,:,0] = momentum[:,:,0]/rho
    u[:,:,1] = momentum[:,:,1]/rho
    return u


def calculate_f_eq(rho: np.ndarray, u: np.ndarray)-> np.ndarray:
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
    f_eq = np.zeros((XMAX,YMAX,9))
    ## D2Q9 implementation
    ux, uy = u[:,:,0], u[:,:,1]

    # First pre compute commonly used terms
    rho_1 = 2*rho/9
    rho_2 = rho/18
    rho_3 = rho/36

    u2 = ux**2 + uy**2
    ux2 = ux**2
    uy2 = uy**2
    uxuy = ux*uy

    # Use the pre-computed terms to compute equillibrium distributions
    f_eq[:,:,0] = rho_1*(2 - 3*u2)
    f_eq[:,:,1] = rho_2*(2 + 6*ux + 9*ux2 - 3*u2)
    f_eq[:,:,2] = rho_2*(2 + 6*uy + 9*uy2 - 3*u2)
    f_eq[:,:,3] = rho_2*(2 - 6*ux + 9*ux2 - 3*u2)
    f_eq[:,:,4] = rho_2*(2 - 6*uy + 9*uy2 - 3*u2)
    f_eq[:,:,5] = rho_3*(1 + 3*(ux + uy) + 9*uxuy +3*u2)
    f_eq[:,:,6] = rho_3*(1 - 3*(ux - uy) - 9*uxuy +3*u2)
    f_eq[:,:,7] = rho_3*(1 - 3*(ux + uy) + 9*uxuy +3*u2)
    f_eq[:,:,8] = rho_3*(1 + 3*(ux - uy) - 9*uxuy +3*u2)

    return f_eq


## Simulation steps
def collision_step(f: np.ndarray, f_eq: np.ndarray, dt=1, tau=1)-> np.ndarray:
    """
    BGK collision step
    :param f: array containing the current population
    :param f_eq: array containing the equillibrium population
    :retunr: population after collision
    """
    return f*(1-dt/tau) + dt/tau*f_eq


def streaming_step(f: np.ndarray)-> np.ndarray:
    # Streaming
    for i in range(9):
        f[:,:,i] = np.roll( np.roll(f[:,:,i], c[0,i], axis=0), c[1,i], axis=1 )
    return f


def BC_solids(f: np.ndarray, solids: np.ndarray[bool])-> np.ndarray:
    b = f[solids,:]
    b = b[:,[0,3,4,1,2,7,8,5,6]]
    f[solids,:] = b
    return f

