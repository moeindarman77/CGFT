function [] = post_proccess(data_path)
%% Data Loading 
X = load(data_path);
%% Preparing Data
U = permute(X.UV(:,1,:,:), [3 4 1 2]);
V = permute(X.UV(:,2,:,:), [3 4 1 2]);

Tau11_model = permute(X.Tau_model(:,1,:,:), [3 4 1 2]);
Tau12_model = permute(X.Tau_model(:,2,:,:), [3 4 1 2]);
Tau22_model = permute(X.Tau_model(:,3,:,:), [3 4 1 2]);

Tau11_FDNS = permute(X.Tau_FDNS(:,1,:,:), [3 4 1 2]);
Tau12_FDNS = permute(X.Tau_FDNS(:,2,:,:), [3 4 1 2]);
Tau22_FDNS = permute(X.Tau_FDNS(:,3,:,:), [3 4 1 2]);

data_size = size(U,3);
%%
for i = 1:data_size
    PTau_FDNS(:,:,i) = energyTransfer_2D_FHIT(U(:,:,i),V(:,:,i),Tau11_FDNS(:,:,i),Tau12_FDNS(:,:,i),Tau22_FDNS(:,:,i));
    PTau_model(:,:,i) = energyTransfer_2D_FHIT(U(:,:,i),V(:,:,i),Tau11_model(:,:,i),Tau12_model(:,:,i),Tau22_model(:,:,i));
    
    %% Calculate the CC for tau11, tau12, tau22 (3 numbers)
    CCTau11(i) = corr2(Tau11_model(:,:,i), Tau11_FDNS(:,:,i));
    CCTau12(i) = corr2(Tau12_model(:,:,i), Tau12_FDNS(:,:,i));
    CCTau22(i) = corr2(Tau22_model(:,:,i), Tau22_FDNS(:,:,i));
    
    %% Calculate CC PTau (1 number)
    PTau_FDNS(:,:,i) = energyTransfer_2D_FHIT(U(:,:,i), V(:,:,i), Tau11_FDNS(:,:,i), Tau12_FDNS(:,:,i), Tau22_FDNS(:,:,i));
    PTau_model(:,:,i) = energyTransfer_2D_FHIT(U(:,:,i), V(:,:,i), Tau11_model(:,:,i), Tau12_model(:,:,i), Tau22_model(:,:,i));

    
    %% Calculate CC(PTau>0) & CC(PTau<0) (2 numbers)
    [CCPTau(i), CCPTauPos(i), CCPTauNeg(i)] = ...
        CCEnergyTransfer_2D_FHIT(U(:,:,i),V(:,:,i),Tau11_FDNS(:,:,i),Tau12_FDNS(:,:,i),Tau22_FDNS(:,:,i), Tau11_model(:,:,i), Tau12_model(:,:,i), Tau22_model(:,:,i));

    %% Calculate Angle averaged spectra for tau11, tau12, tau22 (3 spectra)
    Tau11_model_hat(:,i)  = spectrum_angled_average_2D_FHIT(Tau11_model(:,:,i));
    Tau12_model_hat(:,i)  = spectrum_angled_average_2D_FHIT(Tau12_model(:,:,i));
    Tau22_model_hat(:,i)  = spectrum_angled_average_2D_FHIT(Tau22_model(:,:,i));
    
    Tau11_FDNS_hat(:,i) = spectrum_angled_average_2D_FHIT(Tau11_FDNS(:,:,i));
    Tau12_FDNS_hat(:,i) = spectrum_angled_average_2D_FHIT(Tau12_FDNS(:,:,i));
    Tau22_FDNS_hat(:,i) = spectrum_angled_average_2D_FHIT(Tau22_FDNS(:,:,i));

end

%% Calculating the MEAN and STD for calculated CCs
CCPTau_mean = mean(CCPTau,2);
CCPTau_std = std(CCPTau);

CCPTauPos_mean = mean(CCPTauPos,2);
CCPTauPos_std = std(CCPTauPos);

CCPTauNeg_mean = mean(CCPTauNeg, 2);
CCPTauNeg_std = std(CCPTauNeg);

%% Visualization
% Set up the figure
h1 = figure('Color', 'w', 'Units', 'centimeters', 'Position', [0 0 19 12.5], ...
    'DefaultTextInterpreter', 'latex', ...
    'DefaultAxesTickLabelInterpreter', 'latex', ...
    'DefaultLegendInterpreter', 'latex', ...
    'DefaultColorbarTickLabelInterpreter', 'latex', ...
    'DefaultAxesFontSize', 20, ...
    'DefaultAxesFontName', 'Helvetica', ...
    'DefaultLineMarker', 'o', ...
    'DefaultLineLineWidth', 3,...
    'WindowState','maximized');

% Plotting
loglog(mean(Tau11_FDNS_hat,2),'DisplayName','$\hat{\tau}_{11}^{FDNS}$'); hold on;
loglog(mean(Tau12_FDNS_hat,2),'DisplayName','$\hat{\tau}_{12}^{FDNS}$'); hold on;
loglog(mean(Tau22_FDNS_hat,2),'DisplayName','$\hat{\tau}_{22}^{FDNS}$'); hold on;
loglog(mean(Tau11_model_hat,2),'DisplayName','$\hat{\tau}_{11}^{CNN}$'); hold on;
loglog(mean(Tau12_model_hat,2),'DisplayName','$\hat{\tau}_{12}^{CNN}$'); hold on;
loglog(mean(Tau22_model_hat,2),'DisplayName','$\hat{\tau}_{22}^{CNN}$'); hold on;

% Plot's Label and title
xlabel('$\kappa$')
ylabel("$|{\hat{\tau} }_{ij}(\kappa)|$")
title('Spectrum of $\tau_{ij}$ CNN vs. FDNS ')
legend('Box','off')
xlim([2 100])

plot_name = 'spectrum.pdf'
exportgraphics(h1,plot_name,'Resolution',300,'ContentType','vector')
end
