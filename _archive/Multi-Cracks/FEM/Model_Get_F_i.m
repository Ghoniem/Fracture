function [F_i]=Model_Get_F_i(Model)

NDOF=Model.NDOF;
F_i=zeros(NDOF,1);

for i=1:1:length(Model.Element)

f=Model.Element{i}.f;

if isnan(f); error('Internal Force Value is NaN'); end

for j=1:1:length(Model.Element{i,1}.Nodes)    
Node_num=Model.Element{i,1}.Nodes(j);
NN(2*j-1)=Model.Node{Node_num}.DOFx;  NN(2*j)=Model.Node{Node_num}.DOFy;    
end  

F_i(NN)=F_i(NN)+f;
end

end