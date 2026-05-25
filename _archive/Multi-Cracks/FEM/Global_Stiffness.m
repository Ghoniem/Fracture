function K=Global_Stiffness(Model)  

NDOF=Model.NDOF;
NElement=length(Model.Element);


for i=1:1:NElement
    
NNodeElement=length(Model.Element{i,1}.Nodes);    
NDOFElement=2*NNodeElement;

for j=1:1:NNodeElement   
Node_num=Model.Element{i,1}.Nodes(j);                             
NN(2*j-1)=Model.Node{Node_num}.DOFx;  NN(2*j)=Model.Node{Node_num}.DOFy;    
end    

M=meshgrid(1:NDOFElement)';M2=M';
ii=M(:);
jj=M2(:);
kk=1:NDOFElement^2;
k=Model.Element{i}.K;   
kkk=kk+NDOFElement^2*(i-1);

Ig(kkk)=NN(ii);
Jg(kkk)=NN(jj);
Kg(kkk)=k(:);

end
 
K=sparse(Ig,Jg,Kg,NDOF,NDOF);

end

