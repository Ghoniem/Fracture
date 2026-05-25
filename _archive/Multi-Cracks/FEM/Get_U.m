function U=Get_U(Model)

U=zeros(Model.NDOF,1);                           
for i=1:1:length(Model.Node) 
   U(Model.Node{i,1}.DOFx,1)=Model.Node{i}.Ux; 
   U(Model.Node{i,1}.DOFy,1)=Model.Node{i}.Uy; 
end

end