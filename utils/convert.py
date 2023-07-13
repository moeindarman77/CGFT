import numpy as nnp
import jax.numpy as np
from jax import grad, jit, vmap, pmap
from functools import partial
import glob
import os

def eddyTurnoverTime_2DFHIT(Omega, definition='Enstrophy'):
    """
    Compute eddy turnover time for 2D_FHIT using Omega.
    
    Args:
    A (ndarray): 2D array of Omega U.
    definition (str): Optional string to define eddy turnover time. Default is 'Enstrophy'.
                      Possible values: 'Enstrophy', 'Omega', 'Velocity'
                      
    Returns:
    float: Eddy turnover time.
    """

    eddyTurnoverTime = 1/np.sqrt(np.mean(Omega**2))
    
    return eddyTurnoverTime

def get_last_file(file_path):
    # Get all .mat files in the specified directory
    mat_files = glob.glob(os.path.join(file_path, "*.mat"))
    
    # Extract the integer values from the filenames
    file_numbers = [int(os.path.splitext(os.path.basename(file))[0]) for file in mat_files]
    
    # Find the highest integer value
    if file_numbers:
        last_file_number = max(file_numbers)
        return last_file_number
    else:
        return None

import torch
import torch.fft

def initialize_wavenumbers_2DFHIT(nx, ny, Lx, Ly):
    """
    Initialize the wavenumbers for 2D Forced Homogeneous Isotropic Turbulence (2D-FHIT).
    
    Parameters:
    -----------
    nx : int
        Number of grid points in the x-direction.
    ny : int
        Number of grid points in the y-direction.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.

    Returns:
    --------
    Kx : torch.Tensor
        2D array of wavenumbers in the x-direction.
    Ky : torch.Tensor
        2D array of wavenumbers in the y-direction.
    Ksq : torch.Tensor
        2D array of the square of the wavenumber magnitudes.
    """
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=Lx / nx)
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=Ly / ny)
    Kx, Ky = torch.meshgrid(kx, ky)
    Ksq = (Kx**2 + Ky**2)
    
    return Kx, Ky, Ksq


# @partial(jit, static_argnums=(4,))
# @jit
def Omega2Psi_2DFHIT(Omega, Kx, Ky, Ksq, spectral=False):
    """
    Calculate the stream function from vorticity.

    This function calculates the stream function (Psi) from vorticity (Omega) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input vorticity is in spectral space and returns stream function in
        spectral space. If False (default), assumes input vorticity is in physical space and
        returns stream function in physical space.

    Returns:
    --------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    # if spectral:
    Omega_hat = Omega
    lap_Psi_hat = -Omega_hat
    Psi_hat = lap_Psi_hat / (-Ksq)
    # Psi_hat[0, 0] = 0
    Psi_hat = Psi_hat.at[0,0].set(0)

    return Psi_hat
    # else:
    #     Omega_hat = np.fft.fft2(Omega)
    #     lap_Psi_hat = -Omega_hat
    #     Psi_hat = lap_Psi_hat / (-Ksq)
    #     # Psi_hat[0, 0] = 0
    #     Psi_hat = Psi_hat.at[0,0].set(0)
    #     Psi = np.real(np.fft.ifft2(Psi_hat))

    #     return Psi

# @partial(jit, static_argnums=(4,))
# @jit
def Psi2Omega_2DFHIT(Psi, Kx, Ky, Ksq, spectral=False):
    """
    Calculate the vorticity from the stream function.

    This function calculates the vorticity (Omega) from the stream function (Psi) using the relationship
    that vorticity is the negative Laplacian of the stream function. The function can handle
    both physical and spectral space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    Ksq : numpy.ndarray
        2D array of the square of the wavenumber magnitudes.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns vorticity in
        spectral space. If False (default), assumes input stream function is in physical space and
        returns vorticity in physical space.

    Returns:
    --------
    Omega : numpy.ndarray
        Vorticity (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    # if spectral:
    Psi_hat = Psi
    lap_Psi_hat = (-Ksq) * Psi_hat
    Omega_hat = -lap_Psi_hat

    return Omega_hat
    # else:
    #     Psi_hat = np.fft.fft2(Psi)
    #     lap_Psi_hat = (-Ksq) * Psi_hat
    #     Omega = -np.real(np.fft.ifft2(lap_Psi_hat))

    #     return Omega
    

# @partial(jit, static_argnums=(4,))
# @jit
def Psi2UV_2DFHIT(Psi, Kx, Ky, Ksq, spectral=False):
    """
    Calculate the velocity components U and V from the stream function.

    This function calculates the velocity components U and V from the stream function (Psi)
    using the relationship between them. The function can handle both physical and spectral
    space calculations.

    Parameters:
    -----------
    Psi : numpy.ndarray
        Stream function (2D array) in physical or spectral space, depending on the 'spectral' flag.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input stream function is in spectral space and returns velocity components
        in spectral space. If False (default), assumes input stream function is in physical space and
        returns velocity components in physical space.

    Returns:
    --------
    U, V : tuple of numpy.ndarray
        Velocity components U and V (2D arrays) in physical or spectral space, depending on the 'spectral' flag.

    """
    # if spectral:
    Psi_hat = Psi
    U_hat = 1.j * Ky * Psi_hat
    V_hat = -1.j * Kx * Psi_hat

    return U_hat, V_hat
    # else:
    #     Psi_hat = np.fft.fft2(Psi)
    #     U_hat = 1.j * Ky * Psi_hat
    #     V_hat = -1.j * Kx * Psi_hat
    #     U = np.real(np.fft.ifft2(U_hat))
    #     V = np.real(np.fft.ifft2(V_hat))

    #     return U, V


# @partial(jit, static_argnums=(6,))
# @jit
# def Tau2PiOmega_2DFHIT(Tau11, Tau12, Tau22, Kx, Ky, Ksq, spectral=False):
#     """
#     Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor.

#     Parameters:
#     -----------
#     Tau11 : numpy.ndarray
#         Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
#     Tau12 : numpy.ndarray
#         Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
#     Tau22 : numpy.ndarray
#         Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
#     Kx : numpy.ndarray
#         2D array of wavenumbers in the x-direction.
#     Ky : numpy.ndarray
#         2D array of wavenumbers in the y-direction.
#     spectral : bool, optional
#         If True, assumes input Tau elements are in spectral space and returns PiOmega in spectral space.
#         If False (default), assumes input Tau elements are in physical space and returns PiOmega in physical space.

#     Returns:
#     --------
#     PiOmega : numpy.ndarray
#         PiOmega (2D array) in physical or spectral space, depending on the 'spectral' flag.

#     """
#     if spectral:
#         Tau11_hat = Tau11
#         Tau12_hat = Tau12
#         Tau22_hat = Tau22
#         PiOmega_hat = (Kx * Ky) * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat

#         return PiOmega_hat

#     else:
#         Tau11_hat = nnp.fft.fft2(Tau11)
#         Tau12_hat = nnp.fft.fft2(Tau12)
#         Tau22_hat = nnp.fft.fft2(Tau22)
#         PiOmega_hat = (Kx * Ky) * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat
#         PiOmega = nnp.real(nnp.fft.ifft2(PiOmega_hat))

#         return PiOmega
import torch
import torch.fft

def Tau2PiOmega_2DFHIT(Tau11, Tau12, Tau22, Kx, Ky, Ksq, spectral=False):
    """
    Calculate PiOmega, the curl of the divergence of Tau, where Tau is a 2D symmetric tensor.

    Parameters:
    -----------
    Tau11 : torch.Tensor
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Tau12 : torch.Tensor
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Tau22 : torch.Tensor
        Element (2D array) of the 2D symmetric tensor Tau in physical or spectral space, depending on the 'spectral' flag.
    Kx : torch.Tensor
        2D array of wavenumbers in the x-direction.
    Ky : torch.Tensor
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input Tau elements are in spectral space and returns PiOmega in spectral space.
        If False (default), assumes input Tau elements are in physical space and returns PiOmega in physical space.

    Returns:
    --------
    PiOmega : torch.Tensor
        PiOmega (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    if spectral:
        Tau11_hat = Tau11
        Tau12_hat = Tau12
        Tau22_hat = Tau22
        PiOmega_hat = (Kx * Ky) * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat

        return PiOmega_hat

    else:
        Tau11_hat = torch.fft.fft2(Tau11)
        Tau12_hat = torch.fft.fft2(Tau12)
        Tau22_hat = torch.fft.fft2(Tau22)
        PiOmega_hat = (Kx * Ky) * (Tau11_hat - Tau22_hat) - (Kx * Kx - Ky * Ky) * Tau12_hat
        PiOmega = torch.real(torch.fft.ifft2(PiOmega_hat))

        return PiOmega


def prepare_data_cnn(Psi1_hat, Kx, Ky, Ksq):
    U_hat, V_hat = Psi2UV_2DFHIT(Psi1_hat, Kx, Ky, Ksq)
    U = np.real(np.fft.ifft2(U_hat))
    V = np.real(np.fft.ifft2(V_hat))
    input_data = np.stack((U, V), axis=0)
    return input_data

def postproccess_data_cnn(Tau11CNN, Tau12CNN, Tau22CNN, Kx, Ky, Ksq):
    Tau11CNN_hat = np.fft.fft2(Tau11CNN)
    Tau12CNN_hat = np.fft.fft2(Tau12CNN)
    Tau22CNN_hat = np.fft.fft2(Tau22CNN)
    PiOmega_hat = Tau2PiOmega_2DFHIT(Tau11CNN_hat, Tau12CNN_hat, Tau22CNN_hat, Kx, Ky, Ksq)
    return PiOmega_hat
        
import torch
def UV2Omega_2DFHIT(U, V, Kx, Ky, Ksq, spectral=False):
    """
    Calculate Omega, the curl of the velocity field (U, V).

    Parameters:
    -----------
    U : numpy.ndarray
        2D array of the x-component of the velocity field.
    V : numpy.ndarray
        2D array of the y-component of the velocity field.
    Kx : numpy.ndarray
        2D array of wavenumbers in the x-direction.
    Ky : numpy.ndarray
        2D array of wavenumbers in the y-direction.
    spectral : bool, optional
        If True, assumes input U and V are in spectral space and returns Omega in spectral space.
        If False (default), assumes input U and V are in physical space and returns Omega in physical space.

    Returns:
    --------
    Omega : numpy.ndarray
        Omega (2D array) in physical or spectral space, depending on the 'spectral' flag.

    """
    if spectral:
        U_hat = U
        V_hat = V
        Omega_hat = 1.j * (Kx * V_hat - Ky * U_hat)

        return Omega_hat

    else:
        U_hat = torch.fft.fft2(U)
        V_hat = torch.fft.fft2(V)
        Omega_hat = 1.j * (Kx * V_hat - Ky * U_hat)
        Omega = torch.real(torch.fft.ifft2(Omega_hat))

        return Omega


## ----- Radially averaged spectrum ----- ##
def spectrum_angled_average_2DFHIT(A, spectral = False):
    '''
    Compute the radially/angle-averaged spectrum of a 2D square matrix.

    Parameters
    ----------
    A : numpy.ndarray
        The input 2D square matrix. If `spectral` is False, `A` is in the physical domain; otherwise, it is in the spectral domain.
    spectral : bool, optional
        Whether `A` is in the spectral domain. Default is False.

    Returns
    -------
    spectrum : numpy.ndarray
        The radially/angle-averaged spectrum of `A`.
    wavenumbers : numpy.ndarray
        The corresponding wavenumbers.

    Raises
    ------
    ValueError
        If `A` is not a 2D square matrix or `spectral` is not a boolean.
    '''
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('Input is not a 2D square matrix. Please input a 2D square matrix')
    if not isinstance(spectral, bool):
        raise ValueError('Invalid input for spectral. It should be a boolean value')
    if not np.issubdtype(A.dtype, np.number):
        raise ValueError('Input contains non-numeric values')
        
    nx = A.shape[0]
    L = 2 * np.pi
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    ky = 2 * np.pi * np.fft.fftfreq(nx, d=L/nx)
    (wavenumber_x, wavenumber_y) = np.meshgrid(kx, ky, indexing='ij')
    absolute_wavenumber = np.sqrt(wavenumber_x ** 2 + wavenumber_y ** 2)
    absolute_wavenumber = np.fft.fftshift(absolute_wavenumber)
    
    if not spectral:
        spectral_A = np.fft.fft2(A)
    else:
        spectral_A = A
    spectral_A = np.abs(spectral_A) / nx ** 2
    spectral_A = np.fft.fftshift(spectral_A)
    bin_edges = np.arange(-0.5, nx / 2 + 0.5)
    binnumber = np.digitize(absolute_wavenumber.ravel(), bins=bin_edges)
    spectrum = np.bincount(binnumber, weights=spectral_A.ravel())[1:]
    wavenumbers = np.arange(0, nx // 2 + 1)
    return spectrum, wavenumbers