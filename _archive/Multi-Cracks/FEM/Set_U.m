function Model=Set_U(Model,U)
t1=length(Model.Node);                                    
for i=1:1:t1
Model.Node{i,1}.Ux=U(Model.Node{i,1}.DOFx);
Model.Node{i,1}.Uy=U(Model.Node{i,1}.DOFy);
end

end