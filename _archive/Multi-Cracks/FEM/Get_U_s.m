function [U_s]=Get_U_s(Model)

for i=1:1:length(Model.Constraint)                         
    U_s(i,1)=Model.Constraint{i}.U;
end

end
