clear
close all

load data/GSK2013_0_1uM_matrix.mat;
X = GSK2013_0_1uM_matrix;
% load data/GSK2013_1uM_matrix.mat
% X = GSK2013_1uM_matrix;

[d,n] = size(X);

% observed and unobserved entries
O = X ~=-99999;
U = X ==-99999;

% fill missing entries with mean
Xz = X.*(X>=0);
Xz(U) = mean(mean(X(O)));

% denoise at different ranks
figure(1)
for i=1:5
    Xi = Xz;
    r = 25*i; % assumed rank
    [u,d,v] = svd(Xi);
    s = diag(d);
    s(r:end)=0;
    Xi = u*[diag(s) zeros(length(s),length(d)-length(s))]*v';
    subplot(5,2,2*i-1)
    image(2*Xi); axis normal;
    subplot(5,2,2*i)
    image(2*abs(Xi-Xz)); axis normal;
    e(i) = mean(mean(abs(Xi-Xz)));
    m(i) = max(max(abs(Xi-Xz)));
    p(i) = norm(Xi-Xz,'fro')/prod(size(Xz));
end
    

[e;m;p]

% denoise sqrt at different ranks
figure(2)
Xz = Xz.^(.5);
for i=1:5
    Xi = Xz;
    r = 25*i; % assumed rank
    [u,d,v] = svd(Xi);
    s = diag(d);
    s(r:end)=0;
    Xi = u*[diag(s) zeros(length(s),length(d)-length(s))]*v';
    subplot(5,2,2*i-1)
    image(3*Xi); axis normal;
    subplot(5,2,2*i)
    image(3*abs(Xi-Xz)); axis normal;
    e(i) = mean(mean(abs(Xi-Xz)));
    m(i) = max(max(abs(Xi-Xz)));
    p(i) = norm(Xi-Xz,'fro')/prod(size(Xz));
end
    
[e;m;p]