# -------------------------- CC Energy Transfer ---------------------------------
import numpy as np
from utils.energy_transfer_2d_fhit import energy_transfer_2d_fhit

def cc_energy_transfer_2d_fhit(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS, Tau11Model, Tau12Model, Tau22Model):
    """
    Calculate the correlation coefficient (CC) of energy transfer between the truth (or filtered DNS data) and the model.
    
    Parameters
    ----------
    U, V : numpy.ndarray
        Velocities (NxN matrices).
    Tau11FDNS, Tau12FDNS, Tau22FDNS : numpy.ndarray
        SGS stress of the truth or filtered DNS data (NxN matrices).
    Tau11Model, Tau12Model, Tau22Model : numpy.ndarray
        SGS stress of the model (NxN matrices).

    Returns
    -------
    CCPTau : float
        CC between energy transfer of truth and model.
    CCPTauPos : float
        CC between energy transfer of grid points where P_FDNS > 0 (Positive - diffusion/forward scatter).
    CCPTauNeg : float
        CC between energy transfer of grid points where P_FDNS < 0 (Negative - backscatter).
    """
    
    PTauFDNS = energy_transfer_2d_fhit(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS)
    PTauModel = energy_transfer_2d_fhit(U, V, Tau11Model, Tau12Model, Tau22Model)

    CCPTau = np.corrcoef(PTauFDNS.ravel(), PTauModel.ravel())[0, 1]
    CCPTauPos = np.corrcoef(PTauFDNS[PTauFDNS > 0], PTauModel[PTauFDNS > 0])[0, 1]
    CCPTauNeg = np.corrcoef(PTauFDNS[PTauFDNS < 0], PTauModel[PTauFDNS < 0])[0, 1]
    
    return CCPTau, CCPTauPos, CCPTauNeg