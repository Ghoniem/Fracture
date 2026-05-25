function get_cracks()
global C n_c n_b R_min R_max theta_min ...
         theta_max barrier1 barrier2 barrier_rad
        
%%%% PREPARE CRACK & BARRIER DATA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
a1 = 500;
a2 = 1000;
R_c1=[5000 5000 0];
R_c2=[15000 15000 0];
rng(0,'twister');
phi1=0;
phi2=pi;
for i=1:n_c
    C(i).a=(a2-a1).*rand(1,1) + a1;
    C(i).R_c=(R_c2-R_c1).*rand(1,1) + R_c1;
    C(i).phi=(phi2-phi1).*rand(1,1) + phi1;
end


n_b=200;
R_min=zeros(n_b,1);
R_max=zeros(n_b,1);
theta_min=zeros(n_b,1);
theta_max=zeros(n_b,1);
r_b1=5000;
r_b2=10000;
R_b1(1)=100;
R_b2(1)=18000;
R_b1(2)=100;
R_b2(2)=18000;
            for i=1:n_b
            r_b=(r_b2-r_b1).*rand(1,1) + r_b1 ;   
            R_b(1)=(R_b2(1)-R_b1(1)).*rand(1,1) + R_b1(1);
            R_b(2)=(R_b2(2)-R_b1(2)).*rand(1,1) + R_b1(2);
            barrier1(i)=R_b(1);
            barrier2(i)=R_b(2);
            barrier_rad(i)=r_b;
            R_min(i,1)=norm(R_b)-r_b;
            R_max(i,1)=norm(R_b)+r_b;
            alpha=asin(r_b/norm(R_b));
            theta_b=atan2(R_b(2),R_b(1));
            theta_min(i,1)=theta_b-alpha;
            theta_max(i,1)=theta_b+alpha;
            end


% C(1).a=1000;
% C(1).R_c=[10000 5000 0];
% C(1).phi=0;
% C(2).a=500;
% C(2).phi=0;
% C(2).R_c=[13000 5000 0];
% C(3).a=1000;
% C(3).phi=2*pi/3;
% C(3).R_c=[15000 10000 0];
% C(4).a=500;
% C(4).phi=0;
% C(4).R_c=[5000 15000 0];
% C(5).a=1000;
% C(5).phi=pi/3;
% C(5).R_c=[10000 15000 0];
% C(6).a=800;
% C(6).phi=pi/2;
% C(6).R_c=[15000 15000 0];
% C(7).a=1000;
% C(7).phi=pi/3;
% C(7).R_c=[5000 5000 0];
% C(8).a=500;
% C(8).phi=pi/2;
% C(8).R_c=[10000 5000 0];
% C(9).a=800;
% C(9).phi=0;
% C(9).R_c=[15000 5000 0];
end