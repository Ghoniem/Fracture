function Plot_Results(Model,Show_Deformed,Mag_Factor,Result_type)

NNode=length(Model.Node);

for i=1:1:NNode

X(i,1)=Model.Node{i}.x;
Y(i,1)=Model.Node{i}.y;    
Ux(i,1)=Model.Node{i}.Ux;
Uy(i,1)=Model.Node{i}.Uy;
    
end

figure;

if Show_Deformed==1
X=X+Ux*Mag_Factor;
Y=Y+Uy*Mag_Factor;
end

NElement=length(Model.Element);

for i=1:1:NElement

ex=Model.Element{i}.Material_State.e(1);
ey=Model.Element{i}.Material_State.e(2);
exy=Model.Element{i}.Material_State.e(3);
e1=(ex+ey)/2+sqrt(((ex-ey)/2)^2+exy^2);
e2=(ex+ey)/2-sqrt(((ex-ey)/2)^2+exy^2);

sx=Model.Element{i}.Material_State.s(1);
sy=Model.Element{i}.Material_State.s(2);
sxy=Model.Element{i}.Material_State.s(3);
s1=(sx+sy)/2+sqrt(((sx-sy)/2)^2+sxy^2);
s2=(sx+sy)/2-sqrt(((sx-sy)/2)^2+sxy^2);

    
if Result_type==1    
Result(i,1)=ex;
end
if Result_type==2    
Result(i,1)=ey;
end
if Result_type==3    
Result(i,1)=exy;
end
if Result_type==4    
Result(i,1)=e1;
end
if Result_type==5    
Result(i,1)=e2;
end
if Result_type==6    
Result(i,1)=sx;
end
if Result_type==7    
Result(i,1)=sy;
end
if Result_type==8    
Result(i,1)=sxy;
end
if Result_type==9    
Result(i,1)=s1;
end
if Result_type==10    
Result(i,1)=s2;
end


Triangles(i,1)=Model.Element{i}.Nodes(1);
Triangles(i,2)=Model.Element{i}.Nodes(2);
Triangles(i,3)=Model.Element{i}.Nodes(3);

end

if Result_type==0  %No Contour

patch('Faces',Triangles,'Vertices',[X Y],'FaceColor','none','EdgeColor','green','EdgeAlpha',1);

else  %Other Results
    
patch('Faces',Triangles,'Vertices',[X Y],'FaceVertexCData',Result,'FaceColor','flat','EdgeAlpha',0.05);
colorbar

end

daspect([1 1 1]);

end