function Model=Model_Set_Initial_Conditions(Model)

Initial_Elastic_Plane_Stress_Model_State.e=[0 0 0]';
Initial_Elastic_Plane_Stress_Model_State.s=[0 0 0]';

for i=1:1:length(Model.Node)
Model.Node{i}.Ux=0; 
Model.Node{i}.Uy=0; 
end

for i=1:1:length(Model.Element)

m=Model.Element{i,1}.Material;

if strcmp(Model.Material{m}.Type,'Elastic Plane Stress')
Model.Element{i,1}.Material_State=Initial_Elastic_Plane_Stress_Model_State;
end

end

end