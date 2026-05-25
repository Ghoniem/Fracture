
function [climb,position]=get_motion(i,j)
    global C e_z sig_e sig_amp t n_steps n_b X_intercept Y_intercept n_c ...
    C_C_tol C_C_left C_C_right

        t_max=max(t(i,:));
        sig_tot=-C(i).D(j).sigma+sig_e;
        
%%%%%%%%%%%%%% CALCULATE FORCES AND MOTION %%%%%%%%%%%%%%%%%%%%%%%%
        if j<=C(i).n_d
        C(i).D(j).F_PK=cross(sig_tot*C(i).D(j).b',e_z);
        else
        C(i).D(j).F_PK=cross(sig_tot*C(i).D(j).b',-e_z);
        end
        climb=dot(C(i).D(j).F_PK,C(i).e_p);

        if sig_amp==0
        delr=sign(climb)*C(i).d/n_steps;
        else
        delr=C(i).d*climb/(n_steps*max(sig_amp,t_max));
        end

   for I=1:n_b
        Toughness_L=get_toughness(C(i).D(1).R_p,I);
        Toughness_R=get_toughness(C(i).D(2*C(i).n_d).R_p,I);
      if ((j==1)  && Toughness_L>abs(climb)) || ((j==2*C(i).n_d) && Toughness_R>abs(climb))
                delr=0;
      end
   end
%%%%%%%%%%%%%% CHECK CRACK-CRACK DISTANCES %%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:n_c
    if i~=ii
        R_intercept=[X_intercept(i,ii), Y_intercept(i,ii), 0];
        C_C_left(i,ii)=norm(C(i).D(1).R_p-R_intercept);
        C_C_right(i,ii)=norm(C(i).D(2*C(i).n_d).R_p-R_intercept);
            if (C_C_left(i,ii)<C_C_tol || C_C_right(i,ii)<C_C_tol) && (j==1||j==2*C(i).n_d)
            delr=0;
            end
    end
end
        C(i).D(j).R_p=C(i).D(j).R_p+delr*C(i).e_p;
%         position=dot((C(i).D(j).R_p-C(i).R_L),C(i).e_p);
                position=dot(C(i).D(j).R_p,C(i).e_p);

end