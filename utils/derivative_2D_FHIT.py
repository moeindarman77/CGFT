import torch
# -------------------------- 2D Derivative ---------------------------------
def derivative_2D_FHIT(T, order, Kx, Ky, spectral=False):
    """
    Calculate spatial derivatives for 2D_FHIT in spectral space.
    Boundary conditions are periodic in x and y spatial dimensions
    Length of domain 2*pi

    Input:
    T_hat: Input flow field in spectral space: Square Matrix NxN
    order [orderX, orderY]: Array of order of derivatives in x and y spatial dimensions: [Interger (>=0), Integer (>=0)] 
    Kx, Ky: Kx and Ky values calculated beforehand.

    Output:
    Tderivative_hat: derivative of the flow field T in spectral space: Square Matrix NxN
    """
    if spectral == False:
        T_hat = torch.fft.fft2(T)
    elif spectral == True:
        T_hat = T_hat

    orderX = order[0]
    orderY = order[1]

    # Calculating derivatives in spectral space
    Tderivative_hat = ((1j*Kx)**orderX) * ((1j*Ky)**orderY) * T_hat

    if spectral == False:
        Tderivative = torch.real(torch.fft.ifft2(Tderivative_hat))
        return Tderivative
    elif spectral == True:
        return Tderivative_hat