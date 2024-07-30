%Demo studying the performance of the KPCA-ADDPROD method
%on a real-world dataset, as dimensionality reduction technique before
%clustering with kmeans

%Author: Lynn Houthuys

%Citation: 

%Houthuys L., Suykens J.A.K., "Tensor Learning in Multi-View Kernel PCA", in 
%Proc. of theThe 27th International Conference on Artificial Neural Networks (ICANN)
%, Rhodes, Greece, Oct. 2018, pp. 205-215.

% Dataset : T. Kolenda, L.K. Hansen, J. Larsen, O. Winther, Independent component analysis for
% understanding multimedia content, in: Proceedings of IEEE Workshop on Neural Networks for 
% Signal Processing, 12, 2002, pp. 757–766.

clear all;
rng default;
addpath('MVKPCAutils');

%% Download data
load('Kolenda_withNoise.mat');
N=size(X{1},1);
V=3;
runs=10; %because of the local minima solutions of kmeans

%% Settings - parameters obtained by tuning
nc=3; eta=1;
kernel='RBF_kernel';
load('params');

%% kmeans without dimensionality reduction
perf=zeros(1,V);
for v=1:V
    p=zeros(1,runs);
    for r=1:runs
        idx=kmeans(X{v},nc);
        p(r)=nmi(Y,idx);
    end
    perf(v)=mean(p);
end
disp(['NMI KM: ' num2str(perf(1)) ' ' num2str(perf(2)) ' ' num2str(perf(3))]);

%% KPCA + kmeans
perf=zeros(1,V);
for v=1:V
    p=zeros(1,runs);
    for r=1:runs
        [~,~,et] = kpca_RKM(X{v},kernel,sig_kpca(v),eta,X{v},'eig',nb_pcs_kpca(v));
        idx=kmeans(et,nc);
        p(r)=nmi(Y,idx);
    end
    perf(v)=mean(p);
end
disp(['NMI KPCA + KM: ' num2str(perf(1)) ' ' num2str(perf(2)) ' ' num2str(perf(3))]);

%% KPCA-ADDPROD + kmeans
p=zeros(1,runs);
for r=1:runs
    [~,~,et] = kpca_addprod2(X,kernel,sig_addprod,eta,X,'eig',nb_pcs_addprod,rho_addprod);
    idx=kmeans(et,nc);
    p(r)=nmi(Y,idx);
end
perf=mean(p);

disp(['NMI KPCA-ADDPROD + KM: ' num2str(perf)]);
