function [CCPTau, CCPTauPos, CCPTauNeg] = ...
    CCEnergyTransfer_2D_FHIT(U,V,Tau11FDNS,Tau12FDNS,Tau22FDNS, Tau11Model, Tau12Model, Tau22Model)
%% Correlation coefficient (CC) of Energy transfer of the truth vs model

% Input: Single snapshot (NxN Matrix)
% U,V: Velocities
% Tau11FDNS, Tau12FDNS, Tau22FDNS: SGS stress of the Truth or filtered DNS data
% Tau11Model, Tau12Model, Tau22Model: SGS stress of the Model

% Output: 
% CCPTau: CC between Energy transfer of truth and model
% CCPTauPos: CC between Energy transfer of grid points where P_FDNS > 0 (Positive - diffusion/forward scatter) 
% CCPTauNeg: CC between Energy transfer of grid points where P_FDNS < 0 (Negative - backscatter)

%%

    PTauFDNS = energyTransfer_2D_FHIT(U,V,Tau11FDNS,Tau12FDNS,Tau22FDNS);
    PTauModel = energyTransfer_2D_FHIT(U,V,Tau11Model,Tau12Model,Tau22Model);

    CCPTau = corr2(PTauFDNS,PTauModel);
    CCPTauPos = corr2(PTauFDNS(PTauFDNS>0),PTauModel(PTauFDNS>0));
    CCPTauNeg = corr2(PTauFDNS(PTauFDNS<0),PTauModel(PTauFDNS<0));
    
end