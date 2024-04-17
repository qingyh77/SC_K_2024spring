function [A,B,c,Sit,Rit,Pjt,Qjt] = ICPD_C0(Yt,Wt,Rrank,A0,B0,C0,Si0,Ri0,Pj0,Qj0,lambda,mju)
%% I-CPD function

%%
if nargin<12
    mju =1e-6;
end



Y = Yt;
W = Wt;
Wvec = W(:);
Yvec = Y(:);
[I,J] = size(Y);
W_trans = Wt';

% Y1 = tens2mat(Y,[],1);
% Y2 = tens2mat(Y,[],2);
% Y3 = tens2mat(Y,[],3);
% W1 = tens2mat(W,[],1);
% W2 = tens2mat(W,[],2);
% W3 = tens2mat(W,[],3);

%C = randn(dW,Rrank);
ct = C0;
A = A0;
B = B0;


%% Update A
for ii = 1:I
    sit = lambda * Si0(ii,:) + Yt(ii,:)*diag(Wt(ii,:))*B*diag(ct);
    rit = lambda * Ri0{ii} + diag(ct)* B' * diag(Wt(ii,:)) * B * diag(ct);
    Sit(ii,:) = sit;
    Rit{ii} = rit;

    ati = sit * pinv(rit + mju*eye(Rrank));
    A(ii,:) = ati;

end

%% Update B
for jj = 1:J
    qjt = lambda*Qj0(jj,:)' + diag(ct)* A' *diag(W_trans(jj,:))*Yt(:,jj);
    pjt = lambda*Pj0{jj} + diag(ct)*A' *diag(W_trans(jj,:)) * A *diag(ct);

    Qjt(jj,:) = qjt';
    Pjt{jj} = pjt;

    btj = pinv(pjt+mju*eye(Rrank)) * qjt;
    B(jj,:) = btj';

end

%% Update C
ckBA = kr(B,A);
Sampleindex = find(Wvec);
ckBAsparse = ckBA(Sampleindex,:);
y_omega = Yvec(Sampleindex);

% logg = Wvec==1;
% cksparse = zeros(size(ckBA));
% cksparse(:,logg) = ckBA(:,logg);
ck = pinv(ckBAsparse'*ckBAsparse + mju*eye(Rrank))*ckBAsparse'*y_omega;
c = ck;

end