function [PTau] = energyTransfer_2D_FHIT(U,V,Tau11,Tau12,Tau22)
%% Energy transfer of 2D_FHIT using SGS stress
% input is single snapshot (N x N matrix)

% Inout
% U,V: Velocities
% Tau11, Tau12, Tau22: SGS stress

% Output: 
% PTau: energy transfer

    Ux = derivative_2D_FHIT(U,[1,0],'U');
    Uy = derivative_2D_FHIT(U,[0,1],'U');
    Vx = derivative_2D_FHIT(V,[1,0],'V');
    
    PTau = -(Tau11-Tau22).*Ux - Tau12.*(Uy+Vx);

end