function [X, S, idx] = getUsageTimeSeries()
% USAGE:
%   [X, S] = getUsageTimeSeries()
% OUTPUTS:
%   X: 2075259x7 array of inputs
%           Columns of X:
%           col 1: active power
%           col 2: reactive power
%           col 3: voltage
%           col 4: intensity
%           col 5: sensor measurement 1
%           col 6: sensor measurement 2
%           col 7: sensor measurement 3
%           Warning: Some rows are NaN (if measurements unavailable at that time)
%                    These rows are filled in with interpolated values
%   S: 2075259x7 array of time stamps
%           Columns of S: [year, month, day, hour, minute]
% SOURCE:
%   UCI machine learning repository,
%   Individual household electric power consumption Data Set
%   http:/archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

% ------------------------------------------------------------------------------------------------ %
% ACKNOWLEDGEMENT:
%   Source: http://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption
%   Downloaded and reformatted by Sanjoy Das
% ------------------------------------------------------------------------------------------------ %

filename    = "householdPowerConsumption.txt";
N           = 2075259; % No. of rows (samples)
% FORMAT:
%   Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
% Example:
%   16-12-2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
fileID      = fopen(filename);
rawData     = textscan(fileID,'%{dd-MM-yyyy}D%{hh:mm:ss}T%f%f%f%f%f%f%f',N,'Delimiter',';');
fclose(fileID);
clear file*

% Raw data
D   = rawData{1};
T   = rawData{2};
P   = rawData{3};
Q   = rawData{4};
V   = rawData{5};
I   = rawData{6};
M1  = rawData{7};
M2  = rawData{8};
M3  = rawData{9};

% Convert 9999 (missing values) to NaN
idx = find(P==9999);
P(idx)  = NaN;
Q(idx)  = NaN;
V(idx)  = NaN;
I(idx)  = NaN;
M1(idx) = NaN;
M2(idx) = NaN;
% M3(idx) = NaN; % M3 already has NaN in proper places

% Interpolate missing values
X       = [P Q V I M1 M2 M3];
for k = 1 : size(X,2)
    X(:,k)  = fillnans(X(:,k));
end

% Convert date & time to real arrays
[yr, mo, dy]    = ymd(D);
[hr, mi, ~]     = hms(T);
S       = [yr, mo, dy, hr, mi];



function y = fillnans(x)
nanx = isnan(x);
t    = 1:numel(x);
y    = x;
y(nanx) = interp1(t(~nanx), x(~nanx), t(nanx));


