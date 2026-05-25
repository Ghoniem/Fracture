function [stress]=dynamic_dislocation_field(r,v,t)
            global nu C_T C_L lambda

            K_s=(C_T/v)^2/(2*pi);

            gamma_T=sqrt(1-(v/C_T)^2);
            gamma_L=sqrt(1-(v/C_L)^2);

%%%%%%%%%%%%  Stress   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            x=r(1);y=r(2);
            x_T=(x-v*t)/gamma_T;
            x_L=(x-v*t)/gamma_L;
            r2_T=((x-v*t)/gamma_T)^2+y^2;
            r2_L=((x-v*t)/gamma_L)^2+y^2;

            S_xx=2*K_s*y*(((lambda+2*G)-gamma_L^2*lambda)/(gamma_L*r2_L^2)-(G*(1+gamma_T^2))/(gamma_T*r2_T^2));
            S_yy=2*K_s*y*((lambda-gamma_L^2*(lambda+2*G))/(gamma_L*r2_L^2)+(G*(1+gamma_T^2)/(gamma_T*r2_T^2)));
            S_xy=K_s*((1+gamma_T^2)^2*x_T/(gamma_T^2*r2_T^2)-4*x_L/r2_L^2);



            S_xz=0;
            S_yz=0;
            S_zz=nu*(S_xx+S_yy);
            stress=[S_xx S_xy S_xz;S_xy S_yy S_yz;S_xz S_yz S_zz]; 

%%%%%%%%%%%%  Displacement   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
            u_x=K_d*(atan2(x,y)+x*y/(2*(1-nu)*(r2)));
            u_y=-K_d*(((1-2*nu)/(4*(1-nu)))*log(r2)+(x^2-y^2)/(4*(1-nu)*r2));
            displacement=[u_x,u_y,0];
   end