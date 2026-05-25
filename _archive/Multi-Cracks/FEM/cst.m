function [A B]=cst(V)

x1=V(1,1); y1=V(1,2); 
x2=V(2,1); y2=V(2,2); 
x3=V(3,1); y3=V(3,2); 

A=A_cst(x1,x2,x3,y1,y2,y3);

B=B_cst(x1,x2,x3,y1,y2,y3);

end

function A=A_cst(x1,x2,x3,y1,y2,y3)

x13=x1-x3; y23=y2-y3;

x23=x2-x3; y13=y1-y3;

J=x13*y23-x23*y13;

A=J/2;

end

function B=B_cst(x1,x2,x3,y1,y2,y3)

x21=x2-x1; x32=x3-x2; x13=x1-x3; y12=y1-y2; y23=y2-y3; y31=y3-y1;

x23=x2-x3; y13=y1-y3;

J=x13*y23-x23*y13;

B=[y23 0 y31 0 y12 0; 0 x32 0 x13 0 x21 ; x32 y23 x13 y31 x21 y12]/J;

end
