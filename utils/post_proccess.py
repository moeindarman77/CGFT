# -------------------------- Post Proccess ---------------------------------
import numpy as np
from utils.cc_energy_transfer_2d_fhit import cc_energy_transfer_2d_fhit
from utils.corr2 import corr2

def post_proccess(U, V, Tau11FDNS, Tau12FDNS, Tau22FDNS, Tau11Model, Tau12Model, Tau22Model):

    """
    Perform post-processing on input data and return calculated means and standard deviations for CC values.

    Args:
        U: A 3D array of U values.
        V: A 3D array of V values.
        Tau11FDNS, Tau12FDNS, Tau22FDNS: 3D arrays of FDNS tau values.
        Tau11Model, Tau12Model, Tau22Model: 3D arrays of model tau values.

    Returns:
        output: A dictionary containing the mean and standard deviation of calculated CC values.
    """
    data_size = U.shape[0]

    # Initialize variables
    CCTau11 = np.zeros(data_size)
    CCTau12 = np.zeros(data_size)
    CCTau22 = np.zeros(data_size)
    CCPTau = np.zeros(data_size)
    CCPTauPos = np.zeros(data_size)
    CCPTauNeg = np.zeros(data_size)

    for i in range(data_size):
        # Calculate the CC for tau11, tau12, tau22 (3 numbers)
        CCTau11[i] = corr2(Tau11Model[i], Tau11FDNS[i])
        CCTau12[i] = corr2(Tau12Model[i], Tau12FDNS[i])
        CCTau22[i] = corr2(Tau22Model[i], Tau22FDNS[i])

        # Calculate CC(PTau>0) & CC(PTau<0) (2 numbers)
        CCPTau[i], CCPTauPos[i], CCPTauNeg[i] = cc_energy_transfer_2d_fhit(U[i], V[i], 
                                                                           Tau11FDNS[i], Tau12FDNS[i],Tau22FDNS[i],
                                                                            Tau11Model[i], Tau12Model[i],Tau22Model[i])

    # Calculate the MEAN and STD for calculated CCs
    output = {
        'CCTau11': (np.mean(CCTau11), np.std(CCTau11)),
        'CCTau12': (np.mean(CCTau12), np.std(CCTau12)),
        'CCTau22': (np.mean(CCTau22), np.std(CCTau22)),
        'CCPTau': (np.mean(CCPTau), np.std(CCPTau)),
        'CCPTauPos': (np.mean(CCPTauPos), np.std(CCPTauPos)),
        'CCPTauNeg': (np.mean(CCPTauNeg), np.std(CCPTauNeg))
    }

    return output
