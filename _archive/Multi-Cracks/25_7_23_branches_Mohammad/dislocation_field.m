function [stress,displacement]=dislocation_field(r)
            global nu

            K_s=1/(2*pi*(1-nu));K_d=1/(2*pi);
%%%%%%%%%%%%  Stress   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            r4=norm(r)^4;x=r(1);y=r(2);r2=norm(r)^2;
            S_xx=-K_s*(y*(3*x^2+y^2))/r4;
            S_yy=K_s*(y*(x^2-y^2))/r4;
            S_xy=K_s*(x*(x^2-y^2))/r4;
            S_xz=0;
            S_yz=0;
            S_zz=nu*(S_xx+S_yy);
            stress=[S_xx S_xy S_xz;S_xy S_yy S_yz;S_xz S_yz S_zz]; 

%%%%%%%%%%%%  Displacement   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
            u_x=K_d*(atan2(x,y)+x*y/(2*(1-nu)*(r2)));
            u_y=-K_d*(((1-2*nu)/(4*(1-nu)))*log(r2)+(x^2-y^2)/(4*(1-nu)*r2));
            displacement=[u_x,u_y,0];
   end
        