function GetDislStressField()

global  n_c C S_xx S_yy S_xy Sigma_xx Sigma_yy Sigma_xy n_grid u_x u_y U V ....
         X Y stress n_branch G Length_unit sim_step

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

% Initiate text files where dislocation stress field will be stored at to
% be used in COMSOL FEM conrrection problem
elastic_fields = initialize_files(sim_step);
[X,Y]=generate_grid();
for i_p=1:n_c
    for k_p=1:n_branch  
        for m_p=1:C(i_p).B(k_p).n_modes
            for j_p=1:2*C(i_p).B(k_p).mode(m_p).n_d

                C(i_p).B(k_p).mode(m_p).D(j_p).sigma=zeros(3,3);

                for II=1:n_grid
                    for JJ=1:n_grid 
                        R_pq(1)=C(i_p).B(k_p).mode(m_p).D(j_p).R_p(1)-X(II,JJ);
                        R_pq(2)=C(i_p).B(k_p).mode(m_p).D(j_p).R_p(2)-Y(II,JJ);
                        R_pq(3)=0;
                        r_pq=sqrt(R_pq(1)^2+R_pq(2)^2);
                        if m_p==1
                           QQ=rotation(i_p,j_p,k_p,m_p,0);
                        else 
                           QQ=rotation(i_p,j_p,k_p,m_p,1);
                        end
                        if r_pq<3e-5
                           'self force=0';
                        else
                        r_vec=[X(II,JJ) Y(II,JJ) 0];
                      % sigma_disk=disk_field_pressure(r_vec);
                        [stress,displacement]=dislocation_field(R_pq*QQ);
                        stress=norm(C(i_p).B(k_p).mode(m_p).D(j_p).b)*QQ*stress*QQ';
                        displacement=norm(C(i_p).B(k_p).mode(m_p).D(j_p).b)*displacement*QQ;
                      % S= C(i_p).B(k_p).D(j_p).sigma+stress+sigma_disk;
                        S= C(i_p).B(k_p).mode(m_p).D(j_p).sigma+stress;
                        S_xx(II,JJ)=S(1,1);
                        S_yy(II,JJ)=S(2,2);
                        S_xy(II,JJ)=S(1,2);
                        u_x=displacement(1);
                        u_y=displacement(2);
                        end
                    end
                end

                Sigma_xx=Sigma_xx+S_xx;
                Sigma_yy=Sigma_yy+S_yy;
                Sigma_xy=Sigma_xy+S_xy;
                U=U+u_x;
                V=V+u_y;
              
            end 
        end
    end
end

% Write the dislocation stress file to the text files intitated previously

createFieldFiles(X*Length_unit,Y*Length_unit,U*Length_unit,V*Length_unit,Sigma_xx*G,Sigma_xy*G,Sigma_yy*G,elastic_fields) 