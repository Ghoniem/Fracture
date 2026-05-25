function [stress]=disk_field_point(r,P,D,h)
global Length_unit
            A=2*P/(pi*h); B=A/D; R=D/2;
%%%%%%%%%%%%  Stress   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            x=Length_unit*r(1);y=Length_unit*r(2);r4_plus=(x^2+(R+y)^2)^2;r4_minus=(x^2+(R-y)^2)^2;
            S_xx=B-A*(x^2*(R-y)/r4_minus+x^2*(R+y)/r4_plus);
            S_yy=B-A*((R-y)^3/r4_minus+(R+y)^3/r4_plus);
            S_xy=-A*(x*(R-y)^2/r4_minus-x*(R+y)^2/r4_plus);
            S_xz=0;
            S_yz=0;
            S_zz=(S_xx+S_yy)/3;
            stress=[S_xx S_xy S_xz;S_xy S_yy S_yz;S_xz S_yz S_zz]; 

   end