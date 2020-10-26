clc;
clear;

k = 5;
n0 = 24; 
b = 1;
M = 1:k;
MacroN = 1000;
%% slippage configuration & equal variance
Vdelta1 = sqrt(1/n0);
muX1 = zeros(1,k);
muX1(b) = Vdelta1; 
% Vdelta1 = 0.5;
% Vdelta2 = 0.25;
% Vdelta3 = 0.125;
% Vdelta4 = 0.125/2;
%% MDM1 &MDM2 configuration & equal variance
muX2 = Vdelta1-2*Vdelta1.*(M-1);
%% Var
varX1 = ones(1,k);
varX2 = abs(muX2-Vdelta1)+1;
varX3 = 1./(abs(muX2-Vdelta1)+1);
% varX1 = ones(1,k).*10;  %CONST
% varX2 = 10.*(0.95+0.05.*M); %INC
% varX3 = 10./(0.95+0.05.*M); %DEC

index1 = zeros(1,MacroN);
TotSS1 = zeros(1,MacroN);
Samplesize1 = zeros(MacroN,k);

index2 = zeros(1,MacroN);
TotSS2 = zeros(1,MacroN);
Samplesize2 = zeros(MacroN,k);

index3 = zeros(1,MacroN);
TotSS3 = zeros(1,MacroN);
Samplesize3 = zeros(MacroN,k);

index4 = zeros(1,MacroN);
TotSS4 = zeros(1,MacroN);
Samplesize4 = zeros(MacroN,k);

% parfor i = 1:MacroN
%     [index1(i),TotSS1(i)] = KN(k,n0,Vdelta1,muX1,varX2)  %slippage & equal
% end

% PCS1 = length(find(index1==b))/MacroN;
% AverageTotSS1 = mean(TotSS1);

parfor i = 1:MacroN
    [index2(i),TotSS2(i)] = KNU(k,n0,Vdelta1,muX2,varX1)  %MDM & equal
end

PCS2 = length(find(index2==b))/MacroN;
AverageTotSS2 = mean(TotSS2);
% 
% parfor i = 1:MacroN
%     [index3(i),TotSS3(i)] = KN(k,n0,Vdelta3,muX1,varX2)  %MDM & increasing
% end
% 
% PCS3 = length(find(index3==b))/MacroN;
% AverageTotSS3 = mean(TotSS3);

% parfor i = 1:MacroN
%     [index4(i),TotSS4(i)] = KN(k,n0,Vdelta4,muX1,varX1)  %slippage & decreasing
% end
% PCS4 = length(find(index4==b))/MacroN;
% AverageTotSS4 = mean(TotSS4);
% Tot = [AverageTotSS1;AverageTotSS2;AverageTotSS3];
% PCS = [PCS1;PCS2;PCS3];