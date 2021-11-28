
function [M]=coolMean(M,u)
% faster way of computing feature vector mean in an array
% copyright M. Saquib Sarfraz (KIT), 2019

 u_ =ind2vec(u'); nf=sum(u_,2);
 [~,idx]=sort(u);
 M=M(idx,:);
 
M=cumsum([zeros(1, size(M,2),'single'); M]); 

cnf=cumsum(nf);
nf1=[1;cnf+1];nf1=nf1(1:end-1);
s=[nf1,cnf];


M= M(s(:,2)+1,:)- M(s(:,1),:);
M= M./full(nf);
end