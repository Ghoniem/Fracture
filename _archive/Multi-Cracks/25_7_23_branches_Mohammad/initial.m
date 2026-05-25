
function initial()
global  n_c n_grid C  ...
        S_xx S_yy S_xy Sigma_xx Sigma_yy Sigma_xy  force_ratio ...
         e_x e_y e_z t position u_x u_y U V n_d_min x_cod_up y_cod_up ...
         x_cod_down y_cod_down sig_amp sig_e C_C_tol n_branch nu K_IC
e_x=[1 0 0];
e_y=[0 1 0];
e_z=[0 0 1];
L=zeros(n_c,n_branch);
for i=1:n_c
    for k=1:n_branch
L(i,k)=C(i).B(k).a;
    end
end
a_min=min(L,[],"all");
C_C_tol=a_min/10;

for i=1:n_c
 for k=1:n_branch   
        C(i).B(k).n_d=n_d_min*fix(L(i,k)/a_min);
%         C(i).B(k).n_d=n_d_min;
if C(i).B(k).phi ~= 0 && C(i).B(k).phi ~= pi/2
   C(i).B(k).mode = 'mixed';
else
   C(i).B(k).mode = 'pure';
end

switch(C(i).B(k).mode)

    case 'pure'

        t=ones(n_c,2*C(i).B(k).n_d,n_branch)*sig_amp;
        position=zeros(n_c,2*C(i).B(k).n_d,n_branch);
        x_cod_up=zeros(n_c,2*C(i).B(k).n_d,n_branch);
        y_cod_up=zeros(n_c,2*C(i).B(k).n_d,n_branch);
        x_cod_down=zeros(n_c,2*C(i).B(k).n_d,n_branch);
        y_cod_down=zeros(n_c,2*C(i).B(k).n_d,n_branch);

        C(i).B(k).K = [K_IC,K_IC];
        C(i).B(k).e_p=[cos(C(i).B(k).phi), sin(C(i).B(k).phi), 0];
        C(i).B(k).e_n=[-sin(C(i).B(k).phi), cos(C(i).B(k).phi),0];
        C(i).B(k).R_L=C(i).B(k).R_c-C(i).B(k).a*C(i).B(k).e_p;
        C(i).B(k).R_R=C(i).B(k).R_c+C(i).B(k).a*C(i).B(k).e_p;
        C(i).B(k).d=2*C(i).B(k).a/(2*C(i).B(k).n_d-1);
        C(i).B(k).steps=max(50,C(i).B(k).d/10);
        C(i).B(k).iter_max=10*C(i).B(k).steps;
        % C(i).B(k).b_mag=2*a_min(1,1)*(1-nu)*sig_amp/n_d_min;
        if C(i).B(k).phi == 0
        C(i).B(k).b_mag=2*a_min(1,1)*(1-nu)*sig_e(2,2)/n_d_min;
        else
            C(i).B(k).b_mag=2*a_min(1,1)*(1-nu)*sig_e(1,1)/n_d_min;    
        end
        C(i).B(k).trac=sig_e*C(i).B(k).e_n';
        % C(i).B(k).b_direction=C(i).B(k).trac'/norm(C(i).B(k).trac);
        C(i).B(k).b_direction=[sin(C(i).B(k).phi),cos(C(i).B(k).phi),0];
        C(i).B(k).theta=(atan2(C(i).B(k).trac(2),C(i).B(k).trac(1))+pi/2-C(i).B(k).phi)*0;
        
    for j=1:2*C(i).B(k).n_d     
        C(i).B(k).D(j).R_p=C(i).B(k).R_L+C(i).B(k).d*(j-1)*C(i).B(k).e_p;
        C(i).B(k).D(j).sigma_disk=zeros(3,3);
        % C(i).B(k).D(j).sigma_uniaxial=zeros(3,3);                           % uniaxial
        C(i).B(k).D(j).F_PK=ones(3);
        C(i).B(k).D(j).b=-C(i).B(k).b_mag*C(i).B(k).b_direction;
        C(i).B(k).D(j).recomb=1;
            if j<=C(i).B(k).n_d
            C(i).B(k).D(j).orientation=+1;
            else
            C(i).B(k).D(j).orientation=-1;
            end
    end
            force_ratio=ones(n_c,n_branch);


    case 'mixed'

        t=ones(n_c,2*C(i).B(k).n_d,n_branch,2)*sig_amp;
        position=zeros(n_c,2*C(i).B(k).n_d,n_branch,2);
        x_cod_up=zeros(n_c,2*C(i).B(k).n_d,n_branch,2);
        y_cod_up=zeros(n_c,2*C(i).B(k).n_d,n_branch,2);
        x_cod_down=zeros(n_c,2*C(i).B(k).n_d,n_branch,2);
        y_cod_down=zeros(n_c,2*C(i).B(k).n_d,n_branch,2);

        C(i).B(k).K = [K_IC,K_IC];
        C(i).B(k).e_p=[cos(C(i).B(k).phi), sin(C(i).B(k).phi), 0];
        C(i).B(k).e_n=[-sin(C(i).B(k).phi), cos(C(i).B(k).phi),0];
        C(i).B(k).R_L=C(i).B(k).R_c-C(i).B(k).a*C(i).B(k).e_p;
        C(i).B(k).R_R=C(i).B(k).R_c+C(i).B(k).a*C(i).B(k).e_p;
        C(i).B(k).d=2*C(i).B(k).a/(2*C(i).B(k).n_d-1);
        C(i).B(k).steps=max(50,C(i).B(k).d/10);
        C(i).B(k).iter_max=10*C(i).B(k).steps;
        % C(i).B(k).b_mag=2*a_min(1,1)*(1-nu)*sig_amp/n_d_min;
        T = getTransformationMatrix(i,k);
        C(i).B(k).sig_e_rot = T*sig_e*T';
        C(i).B(k).b_mag1=2*a_min(1,1)*(1-nu)*C(i).B(k).sig_e_rot(1,1)/n_d_min; 
        C(i).B(k).b_mag2=2*a_min(1,1)*(1-nu)*C(i).B(k).sig_e_rot(1,2)/n_d_min; 
        C(i).B(k).trac=C(i).B(k).sig_e_rot*C(i).B(k).e_n';
        % C(i).B(k).trac=sig_e*C(i).B(k).e_n';
        % C(i).B(k).b_direction=C(i).B(k).trac'/norm(C(i).B(k).trac);
        C(i).B(k).b_direction1=[sin(C(i).B(k).phi),cos(C(i).B(k).phi),0];
        C(i).B(k).b_direction2=[cos(C(i).B(k).phi),sin(C(i).B(k).phi),0];
        C(i).B(k).theta=(atan2(C(i).B(k).trac(2),C(i).B(k).trac(1))+pi/2-C(i).B(k).phi)*0;
        
    for j=1:2*C(i).B(k).n_d     
        C(i).B(k).D1(j).R_p=C(i).B(k).R_L+C(i).B(k).d*(j-1)*C(i).B(k).e_p;
        C(i).B(k).D1(j).sigma_disk=zeros(3,3);
        % C(i).B(k).D1(j).sigma_uniaxial=zeros(3,3);                         % uniaxial
        C(i).B(k).D1(j).F_PK=ones(3);
        C(i).B(k).D1(j).b=-C(i).B(k).b_mag1*C(i).B(k).b_direction1;
        C(i).B(k).D1(j).recomb=1;
            if j<=C(i).B(k).n_d
            C(i).B(k).D1(j).orientation=+1;
            else
            C(i).B(k).D1(j).orientation=-1;
            end

        C(i).B(k).D2(j).R_p=C(i).B(k).R_L+C(i).B(k).d*(j-1)*C(i).B(k).e_p;
        C(i).B(k).D2(j).sigma_disk=zeros(3,3);
        % C(i).B(k).D2(j).sigma_uniaxial=zeros(3,3);                         % uniaxial
        C(i).B(k).D2(j).F_PK=ones(3);
        C(i).B(k).D2(j).b=-C(i).B(k).b_mag2*C(i).B(k).b_direction2;
        C(i).B(k).D2(j).recomb=1;
            if j<=C(i).B(k).n_d
            C(i).B(k).D2(j).orientation=+1;
            else
            C(i).B(k).D2(j).orientation=-1;
            end
    end
            force_ratio=ones(n_c,n_branch,2);
end

  end   
end

        S_xx=zeros(n_grid,n_grid);
        S_yy=zeros(n_grid,n_grid);
        S_xy=zeros(n_grid,n_grid);
        Sigma_xx=zeros(n_grid,n_grid);
        Sigma_yy=zeros(n_grid,n_grid);
        Sigma_xy=zeros(n_grid,n_grid);
        u_x=zeros(n_grid,n_grid);
        u_y=zeros(n_grid,n_grid);
        U=zeros(n_grid,n_grid);
        V=zeros(n_grid,n_grid);

end