function [Support]=Get_Support(Model)

for i=1:1:length(Model.Constraint)                         
    Support(i,1)=Model.Constraint{i}.DOF;
end

end