import os
import sys
import h5py
import numpy
import shutil
import configargparse
from loguru import logger
from classy import Class
from scipy.interpolate import interp1d


def primordial_potential_powerspectrum():
    
    """
    Returns an interpolator object which 
    gives powerspectrum in the units of Mpc^3
    """

    k = numpy.logspace(-6, 4, 10000) #laying out the k-space samples
    scale_invariant_powerpsectrum = classconfig['A_s'] * (k/classconfig['k_pivot'])**(classconfig['n_s']-1.0)
    interpolator = interp1d(k, (2*numpy.pi**2/k**3) * scale_invariant_powerpsectrum, fill_value='extrapolate')

    return interpolator


def transfer_primordial_potential_to_cdm(field='d_cdm', redshift=0):
    
    """
    A function which returns an interpolator for 
    transfer function.
    Args:
        field (str): primordial field of interest.
            Default is 'phi'
        redshift (float): redshift of transfer function.
            Default is the configuration redshift
    Returns:
        An intterpolator that takes k value in 1/Mpc and
        returns the corresponding transfer function
        interpolator for the given field.
    """
    
    Tk = cosmo.get_transfer(z=redshift)
    Tk = interp1d(Tk['k (h/Mpc)']*cosmo.h(), Tk[field], fill_value='extrapolate')
    return Tk


def matter_power(redshift=0, filename=None):
    
    """
    Function which returns mattter powerspectrum using
    primordial powerspectrum interpolator and transfer
    function interpolator

    Args:
        redshift (float): redshift of desired powerspectrum. 
            Defaults to the config redshift
        filename (str): filename into which powerspectrum is 
            to be written. Defaults to None.
    Returns:
        A 2D array conttaining k, Pk values
    """
    
    k = numpy.logspace(-2.5 ,1, 1000)
    primordial_power = primordial_potential_powerspectrum()
    Tk = transfer_primordial_potential_to_cdm(redshift=redshift) 
    power = numpy.zeros_like(k)
    
    for i, kval in enumerate(k):
        power[i] = primordial_power(kval)*Tk(kval)**2
    
    if filename:
        numpy.save(filename, numpy.column_stack([k, power]))
    
    return power


def matter_perturbation(phik, kmodulus, redshift=0):
    
    """
    A function that applies CDM transfer function to 
    primordial potential modes generated on a grid
    and evolves them to a partticular redshift

    Args:
        phik (numpy.ndarray): primordial potential 
            on a grided 3D k-space
        kmodulus: The corresponding k-grid
        redshift: The redshift to which transfer
            is desired
    Returns:
        An 3D array containing the transferred
        potential phik
    """

    Tk = transfer_primordial_potential_to_cdm(redshift=redshift) 
    Tk = Tk(kmodulus) 
    phik[i,j,k] *= Tk
    phik[0,0,0] = 0 #skip first element since |k| = 0 for that 
    return phik


def class_powerspectrum(redshift=0):
    
    """
    A helper function that returns the class powerspectrum
    at a given redshift 
    Args:
        redshift (float): The redshift at which the power spectrum
            is desired. Default is config redshift.
    Returns:
        A 2D numpy array with (k, Pk) in Mpc units
    """

    kvalues = numpy.logspace(start=-3, stop=2, num=1000)
    class_powerspectrum = numpy.empty_like(kvalues)
    for i, kvalue in enumerate(kvalues):
        class_powerspectrum[i] = cosmo.pk(kvalue, redshift)
    
    return numpy.column_stack([kvalues, class_powerspectrum])


def deltak(boxsize, gridspacing=1.0, redshift=0, fnl=0):
    """
    A function which samples phik modes on a 3D k-space grid
    to be used fro creating initial condition for N-body
    simulation

    Args:
        boxsize (float): box size in Mpc
        gridspacing (float): spacing of grids; boxsize/gridsize
        redshift (float): redshift at which the initial field is
            to be generated
        fnl (float): value of fnl. Default is 0

    Returns:
        The fourier space field phik

    """
    volume = boxsize**3
    gridsize = int(options.boxsize/options.gridspacing)
    midpoint = int(gridsize/2)
    #Setting up the k-space grid
    kspace = 2 * numpy.pi * numpy.fft.fftfreq(n=gridsize, d=gridspacing).astype(numpy.float64)
    kx, ky, kz = numpy.meshgrid(kspace, kspace, kspace)
    kmodulus = numpy.sqrt(kx**2 + ky**2 + kz**2)[:,:,:midpoint + 1]
    
    phik = numpy.zeros_like(kmodulus, dtype=numpy.complex64)
    #delta = numpy.zeros_like(kmodulus, dtype=numpy.complex64)
    
    #Interpolator object 
    primordial_power = primordial_potential_powerspectrum()
    T = transfer_primordial_potential_to_cdm(redshift=redshift) 
     
    #simulating phik's  
    powerspectrum = primordial_power(kmodulus)
    sdev = numpy.sqrt(powerspectrum*(volume)/2.0)
    real = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
    imag = numpy.random.normal(loc=0, scale=sdev, size=sdev.shape)
    phik = (real + 1j*imag).astype(numpy.complex64)
   
    #N-GenIC type cleaning
    phik[0,0,0] = 0
    phik[midpoint,:,:] = 0
    phik[:,midpoint,:] = 0
    phik[:,:,midpoint] = 0

    phik = hermitianize(phik)
    
    '''
    Phik = numpy.zeros_like(phik)

    for i in range(gridsize):
        print(i)
        for j in range(gridsize):
            for k in range(int(gridsize/2)):
                
                if i == midpoint or j == midpoint or k == midpoint:
                    continue
                if i == 0 and j == 0 and k == 0 :
                    continue

                if k > 0:
                    Phik[i,j,k] = phik[i,j,k]

                else:
                    if i == 0:
                        if j >= midpoint:
                            continue
                        else:
                            jj = gridsize - j
                            Phik[0,j,0] = phik[0,j,0]
                            Phik[0,jj,0] = phik[0,j,0].conjugate()
 
                    else:
                        if i >= midpoint:
                            continue
                        else:
                            ii = gridsize - i
                            jj = gridsize - j
                            if ii == gridsize: ii = 0
                            if jj == gridsize: jj = 0
                            Phik[i,j,k] = phik[i,j,k]
                            Phik[ii,jj,k] = phik[i,j,k].conjugate()
    '''
        
    #Adding non-gaussianity
    phi = numpy.fft.irfftn(phik)
    phi = phi + fnl*(phi**2 - numpy.mean(phi*phi.conjugate()))
    phik = numpy.fft.rfftn(phi)
    phik = hermitianize(phik)
   
    phik = phik*T(kmodulus)
    phik = hermitianize(phik)
    return phik


def displacement(deltak, boxsize, displacement_filename, gridspacing=1.0, redshift=0.0, fnl=0):
    
    """
    Function to compute the displacement field from potential field
    Args:
        phik (ndarray): A 3D array containing the potential field in fourier space
        boxsize (float): Size of the box
        gridspacing (float): spacing of grids
        fnl (float): Value of fnl parameter. Defaults to 0
    Returns:
        displacement field vector for x, y and z
    """

    gridsize = int(boxsize/gridspacing)
    midpoint = int(gridsize/2)
    kspace = 2 * numpy.pi * numpy.fft.fftfreq(n=gridsize, d=gridspacing).astype(numpy.float64)
    kx, ky, kz = numpy.meshgrid(kspace, kspace, kspace[:midpoint+1])
    kmodulus = numpy.sqrt(kx**2 + ky**2 + kz**2)
  
    eps = 1e-32
    phik = (-deltak)/(kmodulus**2 + eps)
    phik[0,0,0] = 0
    phik[phik > 1e20] = 0


    phikx = -1j*kx*phik; phiky = -1j*ky*phik; phikz = -1j*kz*phik


    phikx[0,0,0] = 0; phiky[0,0,0] = 0; phikz[0,0,0] = 0;
    
    phikx = hermitianize(phikx)
    phiky = hermitianize(phiky)
    phikz = hermitianize(phikz)

    numpy.save(os.path.join(options.basedir, options.displacementfilename), phikz)
    
    psix = numpy.fft.irfftn(phikx)
    psiy = numpy.fft.irfftn(phiky)
    psiz = numpy.fft.irfftn(phikz)

    return (psix, psiy, psiz)

def velocities(psix, psiy, psiz):
    """
    Function to calculate velocity field from displacement vector fields
    Args:
        psix (ndarray): displacement field along x
        psiy (ndarray): displacement field along y
        psiz (ndarray): displacement field along z
    Returns:
        tuple containing velocities (vx, vy, vz) for all the particles
    """

    f = cosmo.scale_independent_growth_factor_f(options.redshift)
    h = cosmo.Hubble(options.redshift)*options.speed_of_light
    a = 1./(1. + options.redshift)
    factor = f * a * h /numpy.sqrt(a)
    
    return (psix.flatten()*factor, psiy.flatten()*factor, psiz.flatten()*factor)


def positions(phik, boxsize, displacement_filename, gridspacing=1.0, redshift=0.0, fnl=0.0):
    """
    Function which takes potential field at a redshift
    and returns initial condition position and velocity
    Args:
        phik (ndarray): Potential field at initial
            redshift.
        boxsize (float): Size of the box
        gridspacing (float): Spacing of the grid
    Returns:
        (position, velocity) for the particles
    """
    #gs = 2
    psix, psiy, psiz = displacement(phik, boxsize, displacement_filename, gridspacing, redshift, fnl)
    #psix, psiy, psiz = psix[::gs,::gs,::gs], psiy[::gs,::gs,::gs], psiz[::gs,::gs,::gs]
    psix, psiy, psiz = psix/gridspacing**3, psiy/gridspacing**3, psiz/gridspacing**3
    
    space = numpy.arange(start=0, stop=(boxsize+0), step=gridspacing)
    x, y, z = numpy.meshgrid(space, space, space)
    x += psix; y += psiy; z += psiz

    position = numpy.column_stack([x.flatten(), y.flatten(), z.flatten()])
    velocity = numpy.column_stack(velocities(psix, psiy, psiz))
    position[position<0] +=boxsize 
    position[position>boxsize] -=boxsize
    
    return position, velocity

def write_ics(cosmo, options):
    """
    A function that writes out 
    initial condition file
    """
    icsfilename = os.path.join(options.basedir, options.icsfilename)
    shutil.copy2(options.hdf5filename, icsfilename)
    handle = h5py.File(icsfilename, 'r+')
    delta = deltak(options.boxsize, options.gridspacing, options.redshift, options.fnl)
    position, velocity = positions(delta, options.boxsize, options.displacementfilename, options.gridspacing)
    position = position.astype(numpy.float64)
    velocity = velocity.astype(numpy.float64)
    handle['PartType1']['Coordinates'][:] = position[:]
    handle['PartType1']['Velocities'][:] = velocity[:]
    handle.close()


def hermitianize(x):
    """
    A function that hermitianizes the fourier array
    The vectorized logic is taken from:
    `https://github.com/nualamccullagh/zeldovich-bao/`
    and verfied against hard-coded N-GenIC version.
    """
    gridsize = x.shape[0]
    midpoint = int(gridsize/2)

    x[midpoint+1:, 1:,0]= numpy.conj(numpy.fliplr(numpy.flipud(x[1:midpoint, 1:, 0])))
    x[midpoint+1:,0,0] = numpy.conj(x[midpoint-1:0:-1,0,0])
    x[0,midpoint+1:,0] = numpy.conj(x[0,midpoint-1:0:-1,0])
    
    return x


if __name__ == "__main__":
    

    configfilename = './config1024.ini'
    logger.debug('Using %s'%configfilename)
    config = configargparse.ArgParser(default_config_files=[configfilename])
    config.add('--boxsize', type=int, required=True, help='boxsize')
    config.add('--redshift', type=float, required=True, help='redshift of calaculation')
    config.add('--gridspacing', default=1.0, type=float, help='gridspacing in box')
    config.add('--speed_of_light', type=float, default=3e5, help='gridspacing in box')
    config.add('--fnl', type=float, default=0.0, help="fnl parameter")
    config.add('--seed', type=int, default=42, help="fnl parameter")
    config.add('--logfile', help='log file name')
    config.add('--basedir', type=str, help='base directory of snapshots')
    config.add('--hdf5filename', type=str)
    config.add('--icsfilename', type=str)
    config.add('--displacementfilename', type=str)
    config.add('--linearvkfilename', type=str)
    config.add('--final_condition_redshift', default=2, type=float)
    
    options = config.parse_args()
    

    shutil.copy2(configfilename, os.path.join(options.basedir, os.path.basename(configfilename)))


    config = configargparse.ArgParser(default_config_files=['./class.ini'])
    config.add_argument('--h', default=0.7, type=float)
    config.add_argument('--z_pk', default=0.0, type=float)
    config.add_argument('--output', default='mTk', type=str)
    config.add_argument('--omega_b', default=0.02, type=float)
    config.add_argument('--omega_cdm', default=0.128, type=float)
    config.add_argument('--n_s', default=1.0, type=float)
    config.add_argument('--k_pivot', default=0.05, type=float)
    config.add_argument('--P_k_max_1/Mpc', default=100, type=float)
    config.add_argument('--A_s', default=2.15e-9, type=float)
    config.add_argument('--k_min_tau0', default=4.0, type=float)
    config.add_argument('--k_per_decade_for_pk', default=400, type=int)

    classconfig = vars(config.parse_args())

    numpy.random.seed(options.seed)
    
    cosmo = Class()
    cosmo.set(classconfig)
    cosmo.compute()

    write_ics(cosmo, options)
