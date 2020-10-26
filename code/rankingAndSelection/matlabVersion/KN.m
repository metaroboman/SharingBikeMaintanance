%%%%%%%%%%%%%%%%%%%%
%%% Relicate the result in KN(2001)
%by Zhenxia Cheng
%date: 6 May,2020
%%%%%%%%%%%
function [indexb,TotSS]=KN(k,n0,Vdelta,muX,varX)
%% initilization
% k = 5;
% n0 = 24;
% b = 1;
% M = 1:k;
% % slippage configuration & equal variance
% Vdelta = sqrt(1/n0);
% muX = Vdelta - (M-1).*Vdelta;
% 
% %% MDM configuration & unequal variance
% varX = abs(muX - Vdelta) + 1;
% % varX3 = 1./(abs(muX2-Vdelta1)+1);
I = ones(1,k);
Valpha = 0.05;
h2 = -1/Vdelta*log(2*Valpha/(k-1));
X0 = zeros(n0,k);
for i = 1:k
    X0(:,i) = normrnd(muX(i),sqrt(varX(i)),n0,1);
end
hat_muX = 1./n0.*sum(X0);
Samplesize = ones(1,k).*n0;
r = n0;
barX = hat_muX; % sample mean
for i = 1 : k
    for j = 1 : k
        if  i ~= j && I(i)>0 && I(j)>0 && barX(j) - barX(i) <= -Boundary(Vdelta,h2,varX(i)+varX(i),r)
            I(j) = 0;
        end
    end
end
%% screening
while sum(I) > 1
    for i = 1 : k
        if I(i)>0
            barX(i) = (barX(i)*r+normrnd(muX(i),sqrt(varX(i))))/(r+1);
            Samplesize(i) = Samplesize(i) + 1;
        end
    end
    r = r + 1;
    for i = 1 : k
        for j = 1 : k
            if i~=j && I(i)>0 && I(j)>0 && barX(j) - barX(i) <= -Boundary(Vdelta,h2,varX(i)+varX(j),r)
                I(j) = 0;
            end
        end
    end
end
indexb = find(I>0);
TotSS = sum(Samplesize);
end