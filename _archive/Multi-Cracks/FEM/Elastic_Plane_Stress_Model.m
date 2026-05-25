function [Material_State,D]=Elastic_Plane_Stress_Model(Material,Material_State,e)

E=Material.E;
v=Material.v;

D=E/(1-v^2)*[1 v 0; v 1 0; 0 0 (1-v)/2];

s=D*e;

Material_State.e=e;
Material_State.s=s;


end

