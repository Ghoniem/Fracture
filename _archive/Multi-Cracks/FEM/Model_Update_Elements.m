function Model=Model_Update_Elements(Model,Modeli)

t=Model.Thickness;

for i=1:1:length(Model.Element)

for j=1:1:length(Model.Element{i,1}.Nodes)
    
Node_num=Model.Element{i,1}.Nodes(j);

V(j,1)=Model.Node{Node_num}.x;  V(j,2)=Model.Node{Node_num}.y;

U(j,1)=Model.Node{Node_num}.Ux; U(j,2)=Model.Node{Node_num}.Uy;    
    
end

Element=Modeli.Element{i,1};
Material=Modeli.Material{Element.Material};  

[Element]=Element_Analysis(Element,Material,V,U,t);   

Model.Element{i,1}=Element;

end


end




