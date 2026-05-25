close all; clc; clear all;
format short e

global  n_c C S_xx S_yy S_xy Sigma_xx Sigma_yy Sigma_xy n_grid u_x u_y U V ....
         X Y force_ratio eps sig_e  t position stress x_cod_up ....
        y_cod_up x_cod_down y_cod_down out_iter_max inner_iter_max recomb_L
control_data()
% initial()
inner_iter=0;
%%%%%%%%%%%%%%%%%%%%%  Establish initial distribution of dislocations %%%%
eps=eps*out_iter_max;
for out_iter=1:out_iter_max
    eps=eps/out_iter;
for i_q=1:n_c 
    inner_iter=0;
    while ((force_ratio(i_q) >eps) && (inner_iter<inner_iter_max)) 

        inner_iter=inner_iter+1

    for j_q=1:2*C(i_q).n_d
        C(i_q).D(j_q).sigma=zeros(3,3);
        C(i_q).D(j_q).disp=zeros(1,3);
        cod_up=C(i_q).D(j_q).R_p;
        cod_down=cod_up;

                for i_p=1:n_c 
                   for j_p=1:2*C(i_p).n_d 
                       R_pq=C(i_q).D(j_q).R_p-C(i_p).D(j_p).R_p;
                       QQ=rotation(i_p,j_p);
                       r_pq=norm(R_pq);
                       annihilation=C(i_q).D(j_q).orientation+C(i_p).D(j_p).orientation;
                           if (r_pq<C(i_q).d/recomb_L || r_pq<C(i_p).d/recomb_L) && annihilation==0
                              C(i_q).D(j_q).b=[0 0 0];
                              C(i_p).D(j_p).b=[0 0 0]; 
                           elseif r_pq==0 
                                'self force=0';
                           else
                            [stress,displacement]=dislocation_field(R_pq*QQ);
                            stress=norm(C(i_q).D(j_q).b)*QQ*stress*QQ;
                            C(i_q).D(j_q).sigma= C(i_q).D(j_q).sigma+stress;
                           end
                           C(i_q).D(j_q).recomb=ceil(norm(C(i_q).D(j_q).b));
                           C(i_p).D(j_p).recomb=ceil(norm(C(i_p).D(j_p).b));
                   end
                end

        [t(i_q,j_q),position(i_q,j_q)]=get_motion(i_q,j_q);
        [x_cod_up(i_q,j_q), y_cod_up(i_q,j_q),x_cod_down(i_q,j_q), y_cod_down(i_q,j_q)]=COD(i_q,j_q);
        
    end
        max_climb=max(abs(t(i_q,2:(2*C(i_q).n_d-1))));
                 if C(i_q).n_d==1
                 force_ratio(i_q)=eps;
                 else
        force_ratio(i_q)=max_climb/max(abs(t(i_q,:)));
                 end
            convergence=force_ratio(i_q)
            plot_forces(i_q);
            plot_COD(i_q);

    end

for j=2:2*C(i_q).n_d
    if norm( C(i_q).D(j).b )==0
        x_cod_up(i_q,j)=x_cod_up(i_q,j-1);
        y_cod_up(i_q,j)=y_cod_up(i_q,j-1);
        x_cod_down(i_q,j)=x_cod_down(i_q,j-1);
        y_cod_down(i_q,j)=y_cod_down(i_q,j-1);
    end

end

end


 plot_barriers();

end
%%%%%%%%%%%%%%%%%%%%%%%%%%Stress Field %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i_p=1:n_c
    for j_p=1:2*C(i_q).n_d
        C(i_p).D(j_p).sigma=zeros(3,3);
      for II=1:n_grid
            for JJ=1:n_grid 
                R_pq(1)=C(i_p).D(j_p).R_p(1)-X(II,JJ);
                R_pq(2)=C(i_p).D(j_p).R_p(2)-Y(II,JJ);
                r_pq=sqrt(R_pq(1)^2+R_pq(2)^2);
                QQ=rotation(i_p,j_p);
                 if r_pq==0
                            'self force=0';
                 else
                    [stress,displacement]=dislocation_field(R_pq*QQ);
                    stress=norm(C(i_p).D(j_p).b)*QQ*stress*QQ;
                    displacement=norm(C(i_p).D(j_p).b)*displacement*QQ;
                   S= C(i_p).D(j_p).sigma+stress;
                   S_xx(II,JJ)=S(1,1)+sig_e(1,1);
                   S_yy(II,JJ)=S(2,2)+sig_e(2,2);
                   S_xy(II,JJ)=S(1,2)+sig_e(1,2);
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
            
  plot_field(Sigma_xx,Sigma_yy,Sigma_xy,U,V)
 
