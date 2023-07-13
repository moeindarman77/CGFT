# -------------------------- Energy Transfer ---------------------------------
import numpy as np
from utils.derivative_2d_fhit import derivative_2d_fhit

def energy_transfer_2d_fhit(U, V, Tau11, Tau12, Tau22):
    """
    Energy transfer of 2D_FHIT using SGS stress.
    Input is single snapshot (N x N matrix).
    
    Input:
    U, V: Velocities
    Tau11, Tau12, Tau22: SGS stress
    
    Output:
    PTau: energy transfer
    """
    
    Ux, _ = derivative_2d_fhit(U, [1, 0], 'U')
    Uy, _ = derivative_2d_fhit(U, [0, 1], 'U')
    Vx, _ = derivative_2d_fhit(V, [1, 0], 'V')
    
    PTau = -(Tau11 - Tau22) * Ux - Tau12 * (Uy + Vx)
    
    return PTau