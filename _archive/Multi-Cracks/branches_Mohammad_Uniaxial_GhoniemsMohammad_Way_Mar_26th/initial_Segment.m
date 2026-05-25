function initial_Segment(i,k)


global  C  e_x e_y e_z n_d_in sig_amp sig_e n_branch nu K_IC 


e_x=[1 0 0];
e_y=[0 1 0];
e_z=[0 0 1];

C(i).B(k).n_d=n_d_in;

        if C(i).B(k).phi ~= 0 && C(i).B(k).phi ~= pi/2 || sig_e(1,2) ~=0
           C(i).B(k).case = 'mixed';
           C(i).B(k).n_modes = 2;
           for m=1:C(i).B(k).n_modes
               C(i).B(k).mode(m).n_d=n_d_in;
           end
        else
           C(i).B(k).case = 'pure';
           C(i).B(k).n_modes = 1;
           for m=1:C(i).B(k).n_modes
               C(i).B(k).mode(m).n_d=n_d_in;
           end
        end


        for m=1:1:C(i).B(k).n_modes

            C(i).B(k).mode(m).t=ones(1,2*C(i).B(k).mode(m).n_d)*sig_amp;
            C(i).B(k).mode(m).position=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).x_cod_up=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).y_cod_up=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).x_cod_down=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).y_cod_down=zeros(1,2*C(i).B(k).mode(m).n_d);

            C(i).B(k).K = [K_IC,K_IC];
            C(i).B(k).e_p=[cos(C(i).B(k).phi), sin(C(i).B(k).phi), 0];
            C(i).B(k).e_n=[-sin(C(i).B(k).phi), cos(C(i).B(k).phi),0];
            C(i).B(k).R_L=C(i).B(k).R_c-C(i).B(k).a*C(i).B(k).e_p;
            C(i).B(k).R_R=C(i).B(k).R_c+C(i).B(k).a*C(i).B(k).e_p;
            C(i).B(k).mode(m).d=2*C(i).B(k).a/(2*C(i).B(k).mode(m).n_d-1);
            C(i).B(k).mode(m).steps=max(50,C(i).B(k).mode(m).d/10);
            C(i).B(k).mode(m).iter_max=10*C(i).B(k).mode(m).steps;
            T = getTransformationMatrix(i,k);
            C(i).B(k).sig_e_rot = T*sig_e*T';
            C(i).B(k).mode(m).b_mag=2*C(i).B(k).a*(1-nu)*sig_amp/n_d_in;
            C(i).B(k).trac=sig_e*C(i).B(k).e_n';
            if m==1
               C(i).B(k).mode(m).b_direction=[-sin(C(i).B(k).phi),cos(C(i).B(k).phi),0];
            else
               C(i).B(k).mode(m).b_direction=[cos(C(i).B(k).phi),sin(C(i).B(k).phi),0];
            end
        
        for j=1:2*C(i).B(k).n_d 
              % C(i).B(k).mode(m).D(j).R_p=(C(i).B(k).R_L+C(i).B(k).d*(j-1)*C(i).B(k).e_p);
              % C(i).B(k).mode(m).D(j).sigma_disk=zeros(3,3);
                C(i).B(k).mode(m).D(j).sigma_uniaxial=zeros(3,3);                           %uniaxial
                C(i).B(k).mode(m).D(j).F_PK=ones(1,3);
                C(i).B(k).mode(m).D(j).b=-C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction;
                C(i).B(k).mode(m).D(j).recomb=1;
                if j<=C(i).B(k).n_d
                   C(i).B(k).mode(m).D(j).orientation=+1;
                   if j==1
                      C(i).B(k).mode(m).D(j).R_p = C(i).B(k).R_L;
                   else
                      C(i).B(k).mode(m).D(j).R_p = C(i).B(k).R_L+C(i).B(k).mode(m).d*C(i).B(k).e_p;
                   end
                else
                   C(i).B(k).mode(m).D(j).orientation=-1;
                   if j==C(i).B(k).mode(m).n_d+1
                      C(i).B(k).mode(m).D(j).R_p = C(i).B(k).R_R-C(i).B(k).mode(m).d*C(i).B(k).e_p;
                   else
                      C(i).B(k).mode(m).D(j).R_p = C(i).B(k).R_R;
                   end
                end
            C(i).B(k).mode(m).D(j).flag=0; 

        end

            C(i).B(k).mode(m).force_ratio=ones(1,1);
            C(i).B(k).mode(m).PK_force=ones(1,1);
            C(i).B(k).mode(m).annihilation_flag=0;

        end
        n_branch=n_branch+1;
end
