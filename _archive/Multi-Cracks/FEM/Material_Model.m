function [Material_State,D]=Material_Model(Material,Material_State,e)

%Elastic plane stress material
%-----------------------------

if strcmp(Material.Type,'Elastic Plane Stress')   

[Material_State,D]=Elastic_Plane_Stress_Model(Material,Material_State,e);

end


end

