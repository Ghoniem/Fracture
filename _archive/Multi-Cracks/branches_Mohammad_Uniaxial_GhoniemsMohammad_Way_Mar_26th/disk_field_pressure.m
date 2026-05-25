function [stress]=disk_field_pressure(r_vec)
global Length_unit D alpha q G
            A=-2*q/(G*pi); R=D/2;n=50;
%%%%%%%%%%%%  Stress   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            x=Length_unit*r_vec(1);y=Length_unit*r_vec(2);
            r=sqrt(x^2+y^2);theta=-atan2(y,x)+pi/2;
            S_r=alpha;
            S_theta=alpha;
            tau=0;
            for i=1:n
                temp1=(1-(1-1/i)*(r/R)^2)*(r/R)^(2*i-2);
                temp2=(1-(1+1/i)*(r/R)^2)*(r/R)^(2*i-2);
                S_r=S_r+temp1*sin(2*i*alpha)*cos(2*i*theta);
                S_theta=S_theta-temp2*sin(2*i*alpha)*cos(2*i*theta);
                tau=tau+(1-(r/R)^2)*(r/R)^(2*i-2)*sin(2*i*alpha)*sin(2*i*theta);
            end
            S_r=A*S_r;
            S_theta=A*S_theta;
            tau=A*tau;
            
            C1=(S_r+S_theta)/2;
            C2=(S_r-S_theta)/2;
            S_yy= C1+C2*cos(2*theta)+tau*sin(2*theta);
            S_xx= C1-C2*cos(2*theta)-tau*sin(2*theta);
            S_xy= -C2*sin(2*theta)+tau*cos(2*theta);
            S_xz=0;
            S_yz=0;
            S_zz=(S_xx+S_yy)/3;
%             S_xx=S_r;S_yy=S_theta;S_xy=tau;
            stress=[S_xx S_xy S_xz;S_xy S_yy S_yz;S_xz S_yz S_zz]; 

   end