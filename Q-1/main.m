% ========================= NEW BLOCK ========================
%          READ TRAINING, TEST, & VALIDATION IMAGES 
%  THIS BLOCK MAY BE IMPLEMENTED IN PYTORCH WITHOUT CHANGES
% ============================================================
% WARNING: YOU WILL GET DIFFERENT RESULTS EACH TIME!
% SEED THE RANDOM NUMBER GENERATOR TO RECREATE RESULTS!

addpath('weatherDataset');
ratio       = [.85, .15];
[IMAGES, imgSize, nClasses, IMGFILES]   ...
            = readImages(ratio);
Xtrain      = IMAGES.Xtrain;
Ytrain      = IMAGES.Ytrain;
Xvalid      = IMAGES.Xvalid;
Yvalid      = IMAGES.Yvalid;
Ytrain      = categorical(Ytrain);
Yvalid      = categorical(Yvalid);
clear IMAGES ratio
rmpath('weatherDataset');

% OPTIONALLY SAVE DATA (SUITABLE FORMAT) IN FILE xxxx FOR PYTORCH

% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%           DEFINE DEEP NEURAL NETWORK ARCHITECTURE 
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
% DEFINE LAYERS
% You may add new conv+batchNorm+reLu+maxPool layers, 
% or insert dropout layer(s);
% or tweak the constants of the layer(s)
% BE PATIENT; USE INTUITION

inpLyr          = imageInputLayer(imgSize,'Name','InpLyr');

convLyr1        = convolution2dLayer(5,20,'Stride',1,'Padding','same','Name','ConvLyr1');
batchNormLyr1   = batchNormalizationLayer('Name','BatchNormLyr1');
reluLyr1        = reluLayer('Name','ReLuLyr1');
maxPoolLyr1     = maxPooling2dLayer(2,'Stride',2,'Name','MaxPoolLyr1');

fullconLyr      = fullyConnectedLayer(nClasses,'name','FullConLyr');
softmaxLyr      = softmaxLayer('name','SoftMaxLyr');
classifyLyr     = classificationLayer('Name','ClassifyLyr');

layers = [  inpLyr
            convLyr1
            batchNormLyr1
            reluLyr1
            maxPoolLyr1
            fullconLyr
            softmaxLyr
            classifyLyr ];
clear *Lyr*
% PLOT LAYERS
lgraph = layerGraph(layers);
analyzeNetwork(lgraph);
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                  TRAIN DEEP NEURAL NETWORK 
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
% OPTIONALLY LOAD DATA FROM FILE xxxx FOR PYTORCH

% SET TRAINING OPTIONS
% You may tweak:    momentum, initialLearnRate, learnRateSchedule,
%                   learnRateSchedule, learnRateDropFactor, 
%                   MiniBatchSize, etc.  
% BE PATIENT; USE INTUITION
options = trainingOptions(              ...
            'sgdm',                     ...
            'MaxEpochs', 60,            ...
            'shuffle', 'every-epoch',   ...
            'validationData', {Xvalid, Yvalid}, ...
            'validationFrequency', 10,  ...
            'verbose', true(),          ...
            'Plots', 'training-progress' );
% TRAIN NETWORK
neuralNet = trainNetwork(Xtrain, Ytrain, layers, options);
% ======================= END OF BLOCK =======================

% ========================= NEW BLOCK ========================
%                  TEST DEEP NEURAL NETWORK 
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
[correct1, wrong1]  = testNN(neuralNet, Xtrain, Ytrain, "TRAINING");
[correct2, wrong2]  = testNN(neuralNet, Xvalid, Yvalid, "VALIDATION");
% ============================================================

% ========================= NEW BLOCK ========================
%                        FIND YOUR SCORE 
% ============================================================
trnAcc      = 100*correct1/(correct1 + wrong1);  % 60% - 85%
valAcc      = 100*correct2/(correct2 + wrong2);  % 60% - 80%
trnScale    = min(max(trnAcc-60, 0), 25)/25;     % 0 - 1
valScale    = min(max(valAcc-60, 0), 20)/20;     % 0 - 1
score       = 4 + 3*trnScale + 3*valScale;       % 4.0 - 10.0
fprintf(" YOU SCORE %3.1f POINTS OUT OF 10 \n", score);
% ============================================================

% delete(findall(0));  % Deletes all training progress windows

% ====================== NEW FUNCTION ========================
%                  TEST DEEP NEURAL NETWORK 
%    ---- THIS BLOCK MUST BE IMPLEMENTED IN PYTORCH ----
% ============================================================
function [correct, wrong] = testNN(neuralNet, X, Y, s)
Ypred       = classify(neuralNet, X);
correct     = sum(Ypred == Y);
total       = numel(Y);
wrong       = total - correct; 
if ~exist('s','var')
    s = " ";
end
fprintf(strcat("\n NEURAL NETWORK PERFORMANCE WITH ", s, " DATA \n"));
fprintf("Correct: %4d/%4d (%5.2f%%);\n Wrong : %4d/%4d (%5.2f%%)\n", ...
          correct, total, 100*correct/total, wrong, total, 100*wrong/total);
fprintf("-------------------------------------------------------\n");
end
% ===================== END OF FUNCTION ======================

