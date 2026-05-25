function [Element]=Element_Analysis(Element,Material,V,U,t)   

Element_Type=Element.Type;
Material_State=Element.Material_State;
if strcmp(Element_Type,'cst')  
    
[A, B]=cst(V);

Ux1=U(1,1); Uy1=U(1,2);
Ux2=U(2,1); Uy2=U(2,2);
Ux3=U(3,1); Uy3=U(3,2);

e=B*[Ux1 Uy1 Ux2 Uy2 Ux3 Uy3]';

[Material_State,D]=Material_Model(Material,Material_State,e);

K=A*t*B'*D*B;
s=Material_State.s;
f=A*t*B'*s;

Element.A=A;
Element.B=B;
Element.D=D;
Element.K=K;
Element.f=f;
Element.Material_State=Material_State;

end


end

