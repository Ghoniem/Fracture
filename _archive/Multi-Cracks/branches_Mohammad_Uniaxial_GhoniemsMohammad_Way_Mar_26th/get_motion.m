
function [climb,position]=get_motion(i,j,k,m)

global C e_z sig_amp n_steps n_b X_intercept Y_intercept n_c ...
       C_C_tol C_C_left C_C_right resistance recomb_L 

t_max=max(C(i).B(k).mode(m).t,[],"all");

% sig_tot=-C(i).B(k).D(j).sigma+C(i).B(k).D(j).sigma_disk;
sig_tot=-C(i).B(k).mode(m).D(j).sigma+C(i).B(k).mode(m).D(j).sigma_uniaxial;     %uniaxial

%%%%%%%%%%%%%% CALCULATE FORCES AND MOTION %%%%%%%%%%%%%%%%%%%%%%%%
if j<=C(i).B(k).mode(m).n_d
   C(i).B(k).mode(m).D(j).F_PK=cross(sig_tot*C(i).B(k).mode(m).D(j).b',e_z);
   if C(i).B(k).mode(m).D(j).flag==1
      C(i).B(k).mode(m).D(j).F_PK=zeros(1,3);
   end
else
   C(i).B(k).mode(m).D(j).F_PK=cross(sig_tot*C(i).B(k).mode(m).D(j).b',-e_z);
   if C(i).B(k).mode(m).D(j).flag==1
      C(i).B(k).mode(m).D(j).F_PK=zeros(1,3);
   end
end
        
climb=dot(C(i).B(k).mode(m).D(j).F_PK,C(i).B(k).e_p);

if abs(climb)<resistance
   climb=0;
else
   climb=sign(climb)*(abs(climb)-resistance);
end

if sig_amp==0
   delr=sign(climb)*C(i).B(k).mode(m).d/n_steps;
else
   delr=C(i).B(k).mode(m).d*climb/(n_steps*max(sig_amp,t_max));
end

% Ensure the leading dipoles are stationary
if (j==1)  || (j==2*C(i).B(k).mode(m).n_d)
   delr=0;
end

if norm(C(i).B(k).mode(m).D(j).R_p-C(i).B(k).R_R) < C(i).B(k).mode(m).d/recomb_L...
        && C(i).B(k).mode(m).D(j).orientation== -1 || C(i).B(k).mode(m).D(j).R_p(1) > C(i).B(k).R_R(1)...
        && C(i).B(k).mode(m).D(j).orientation== -1 || C(i).B(k).mode(m).D(j).R_p(2) > C(i).B(k).R_R(2) && C(i).B(k).mode(m).D(j).orientation== -1
   delr=0;
   C(i).B(k).mode(m).D(j).R_p=C(i).B(k).R_R;
    
   if j~=1 && j~=2*C(i).B(k).mode(m).n_d
      climb=0;
      delr=0;
   end
end
  
if norm(C(i).B(k).mode(m).D(j).R_p-C(i).B(k).R_L) < C(i).B(k).mode(m).d/recomb_L...
        && C(i).B(k).mode(m).D(j).orientation== +1 || C(i).B(k).mode(m).D(j).R_p(1) < C(i).B(k).R_L(1)...
        && C(i).B(k).mode(m).D(j).orientation== +1 || C(i).B(k).mode(m).D(j).R_p(2) < C(i).B(k).R_L(2) && C(i).B(k).mode(m).D(j).orientation== +1
       
   delr=0;
   C(i).B(k).mode(m).D(j).R_p=C(i).B(k).R_L;

   if j~=1 && j~=2*C(i).B(k).n_d
      climb=0;
      delr=0;
   end
end

for I=1:n_b
    Toughness_L=get_toughness(C(i).B(k).mode(m).D(1).R_p,I);
    Toughness_R=get_toughness(C(i).B(k).mode(m).D(2*C(i).B(k).mode(m).n_d).R_p,I);
    if ((j==1)  && Toughness_L>abs(climb)) || ((j==2*C(i).B(k).mode(m).n_d) && Toughness_R>abs(climb))
       delr=0;
    end
end

%%%%%%%%%%%%%% CHECK CRACK-CRACK DISTANCES %%%%%%%%%%%%%%%%%%%%%%%%%
for ii=1:n_c
    if i~=ii
       R_intercept=[X_intercept(i,ii), Y_intercept(i,ii), 0];
       C_C_left(i,ii)=norm(C(i).B(k).D(1).R_p-R_intercept);
       C_C_right(i,ii)=norm(C(i).B(k).D(2*C(i).B(k).n_d).R_p-R_intercept);
       if (C_C_left(i,ii)<C_C_tol || C_C_right(i,ii)<C_C_tol) && (j==1||j==2*C(i).B(k).n_d)
          delr=0;
       end
    end
end
  
C(i).B(k).mode(m).D(j).R_p=C(i).B(k).mode(m).D(j).R_p+delr*C(i).B(k).e_p;
position=dot(C(i).B(k).mode(m).D(j).R_p,C(i).B(k).e_p);

             
end