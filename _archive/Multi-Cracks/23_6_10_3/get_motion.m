
function [climb,position]=get_motion(i,j)
    global C e_z sig_e sig_amp t n_steps
    t_max=max(t(i,:));
    sig_tot=-C(i).D(j).sigma+sig_e;

        if j<=C(i).n_d
        C(i).D(j).F_PK=cross(sig_tot*C(i).D(j).b',e_z);
        else
        C(i).D(j).F_PK=cross(sig_tot*C(i).D(j).b',-e_z);
        end

    climb=dot(C(i).D(j).F_PK,C(i).e_p);
    delr=C(i).d*climb/(n_steps*max(sig_amp,t_max));


        if sig_amp==0
            delr=sign(climb)*C(i).d/n_steps;
        end

Toughness_L=get_toughness(C(i).D(1).R_p);
Toughness_R=get_toughness(C(i).D(2*C(i).n_d).R_p);

  if ((j==1)  && Toughness_L>abs(climb)) || ((j==2*C(i).n_d) && Toughness_R>abs(climb))
            delr=0;
  end
        
%   if (j==1)  || (j==2*C(i).n_d)
%             delr=0;
%   end

    C(i).D(j).R_p=C(i).D(j).R_p+delr*C(i).e_p;
        position=dot((C(i).D(j).R_p-C(i).R_L),C(i).e_p);
end