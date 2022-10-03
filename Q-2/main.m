% Power consumption forecasting
% There are 7 attributes per time instance 't' (minute)
% The NN inputs are real data:          
%                                       x(t-n  ,1:7)
%                                       x(t-n+1,1:7)
%                                           ...
%                                       x(t-1  ,1:7)
%                                       x(t    ,1:7)
% The NN output is predicted values:    x(t+1  ,1)   = y(t+1)
% ========================= NEW BLOCK ========================
%          READ TRAINING, TEST, & VALIDATION IMAGES 
%  THIS BLOCK MAY BE IMPLEMENTED IN PYTORCH WITHOUT CHANGES
% ============================================================
clear, clc; close all; 
seed = 1;                   % DO NOT CHANGE THESE VALUES
rng(seed);
Ntrain          = 5000;     % DO NOT CHANGE THESE VALUES
Nvalid          = 2000;     % DO NOT CHANGE THESE VALUES

% ASSEMBLE TRAINING AND VALIDATION DATA SETS
Ntot            = Ntrain + Nvalid;
[Data, ~]       = getUsageTimeSeries();
Data            = Data(1:Ntot+1, :)';          
numFields       = size(Data,1);

% SCALE DATA
maxData         = max(Data,[],2);
minData         = min(Data,[],2);
dsize           = [1,Ntot+1];
Data            = Data - repmat(minData,dsize);
Data            = Data./(repmat(maxData,dsize) - repmat(minData,dsize));

% DIVIDE INTO TRAINING & VALIDATION
Xtrain          = Data(:,   1 : Ntrain);
Ytrain          = Data(1:3, 2 : Ntrain+1);
Xvalid          = Data(:,   Ntrain+1 : Ntot);
Yvalid          = Data(1:3, Ntrain+2 : Ntot+1);

% OPTIONALLY SAVE DATA (SUITABLE FORMAT) IN FILE xxxx FOR PYTORCH

% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%           DEFINE DEEP NEURAL NETWORK ARCHITECTURE 
%              YOU CAN MAKE CHANGE TO THIS BLOCK
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
% SPECIFY LAYERS
% YOU CAN ADD FULLY-CONNECTED, RELU, DROPOUT LAYERS, 
% CHANGE LSTM LAYER TO BILSTM, ETC. TO IMPROVE PERFORMANCE
seqInLyr        = sequenceInputLayer(numFields,'Name','Seq-Input');
numHiddenUnits  = 100;
lstmLyr         = lstmLayer(numHiddenUnits,'Name','LSTM');  % MAY CHANGE TO BILSTM TO IMPROVE PERFORMANCE!
numOutputs      = 3;
fullConnLyr     = fullyConnectedLayer(numOutputs,'Name','Full-Conn');
regressLyr      = regressionLayer('Name','MSE-Regress');

% BUILD MULTI-LAYERED NEURAL NETWORK
layers = [  seqInLyr
            lstmLyr
            fullConnLyr
            regressLyr ];
clear *Lyr*

% SHOW MULTI-LAYERED NEURAL NETWORK
analyzeNetwork(layers);
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                  TRAIN DEEP NEURAL NETWORK 
%              YOU CAN MAKE CHANGES TO THIS BLOCK
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
% SET TRAINING OPTIONS
% YOU CAN MODIFY THE TRAINING OPTIONS TO GET BETTER PERFORMANCE
% "ADAM" DOES BETTER THAN THAN "SGDM" WITH RECURRENT NETWORKS
options = trainingOptions(              ...
            'adam',                     ...
            'MaxEpochs', 300,           ...
            'validationData', {Xvalid, Yvalid},   ...
            'validationFrequency', 5,   ...
            'verbose', true(),          ...
            'Plots', 'training-progress' );

% OPTIONALLY LOAD DATA FROM FILE xxxx FOR PYTORCH

% TRAIN NETWORK
neuralNet   = trainNetwork(Xtrain, Ytrain, layers, options);
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                  TEST DEEP NEURAL NETWORK 
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
% RUN NEURAL NETWORK 
Ypred      = predict(neuralNet, Xvalid);
errs       = Ypred - Yvalid;
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                     UNSCALE & PLOT RESULTS
%  THIS BLOCK MAY BE IMPLEMENTED IN PYTORCH WITHOUT CHANGES
% ============================================================
% RESTORE SCALE
maxData(4:end)  = [];
minData(4:end)  = [];
dsize           = [1,Nvalid];
Zvalid          = Yvalid.*(repmat(maxData,dsize) - repmat(minData,dsize));
Zvalid          = Zvalid + repmat(minData,dsize);
Zpred           = Ypred .*(repmat(maxData,dsize) - repmat(minData,dsize));
Zpred           = Zpred  + repmat(minData,dsize);
% PLOT
Ylabels     = {'Active Power'; 'Reactive Power'; 'Voltage'};
for i = 1 : 3
    subplot(3,1,i); 
    plot(1:Nvalid, Zvalid(i,:), 'b', 1:Nvalid, Zpred(i,:), 'r');
    grid on;
    ylabel(Ylabels{i});
    legend("real","predicted");
end
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                  COMPUTE ERRORS & YOUR SCORE 
%  THIS BLOCK MAY BE IMPLEMENTED IN PYTORCH WITHOUT CHANGES
% ============================================================
% COMPUTE ERRORS
fprintf("\n Errors: [");
mse     = NaN(3,1); 
for i = 1 : 3
	mse(i)      = sum(errs(i,:).^2)/Nvalid;
    fprintf(" %6.5f ", sqrt(mse(i)));
end
rmse    = sqrt(sum(mse));   
fprintf("]; Total Error %7.6f.\n", rmse);
minrmse     = 0.024;   % highest is 0.018613 
maxrmse     = 0.107;   % default is 0.102571
pnlty       = (rmse - minrmse)/(maxrmse - minrmse);
if pnlty > 1
    pnlty   = 1;
end
score       =   10*(1 - pnlty.^2); 
fprintf("YOU SCORE %4.1f POINTS OUT OF 10\n", score);
% ======================= END OF BLOCK =======================

% delete(findall(0));