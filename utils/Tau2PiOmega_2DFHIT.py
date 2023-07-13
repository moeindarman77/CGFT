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