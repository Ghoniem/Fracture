function Dislocations_Distribution(flag)

global  n_c C eps sig_e stress max_iterations recomb_L n_branch G Eeff... 
         Length_unit n_d_min n_d_max

for i_q=1:n_c 
     for M=1:n_d_max
        for k_q=1:n_branch
            for m_q=1:C(i_q).B(k_q).n_modes
                      C(i_q).B(k_q).mode(m_q).reset_flag=1;
                while C(i_q).B(k_q).mode(m_q).reset_flag==1
                      C(i_q).B(k_q).mode(m_q).reset_flag=0;

% function to update the crack segment parameters as we add dislocation
% dipoles

update();
% flag 
rsc=0;
PKtemp = ones;

iter=0;
max_iterations=C(i_q).B(k_q).mode(m_q).iter_max*C(i_q).B(k_q).n_d;

while PKtemp > eps

    iter=iter+1;

      for j_q=1:2*C(i_q).B(k_q).mode(m_q).n_d
          C(i_q).B(k_q).mode(m_q).D(j_q).sigma=zeros(3,3);

        % C(i_q).B(k_q).mode(m_q).D(j_q).sigma_disk=disk_field_pressure(C(i_q).B(k_q).D(j_q).R_p);
        % if the flag == 1 the applied stresses on each dislocation is imported from
        % the COMSOL FEM correction problem otherwise the applied stresses
        % will be what ever initial stress tensor specified at the
        % control function

        if flag==0
          C(i_q).B(k_q).mode(m_q).D(j_q).sigma_uniaxial=sig_e;            %uniaxial
        else
          C(i_q).B(k_q).mode(m_q).D(j_q).sigma_uniaxial=C(i_q).B(k_q).mode(m_q).D(j_q).Corrected_Stress;
        end
          C(i_q).B(k_q).mode(m_q).D(j_q).disp=zeros(1,3);
          cod_up=C(i_q).B(k_q).mode(m_q).D(j_q).R_p;
          cod_down=cod_up;

          for i_p=1:n_c 
              for k_p=1:n_branch 
                  for m_p=1:C(i_p).B(k_p).n_modes
                      
                      % Exclude interactions between two modes of the same
                      % crack segment

                      if i_p==i_q && k_p==k_q && m_p~=m_q
                         continue
                      end

                      for j_p=1:2*C(i_p).B(k_p).mode(m_p).n_d 

                           % Rotation matrix for mode I and mode II (Edge
                           % climb and edge glide)
                           if m_p==1
                           QQ=rotation(i_p,j_p,k_p,m_p,0);
                           else
                           QQ=rotation(i_p,j_p,k_p,m_p,1);
                           end

                           R_pq=C(i_q).B(k_q).mode(m_q).D(j_q).R_p-C(i_p).B(k_p).mode(m_p).D(j_p).R_p;
                           r_pq=norm(R_pq);
                           annihilation=C(i_q).B(k_q).mode(m_q).D(j_q).orientation+C(i_p).B(k_p).mode(m_p).D(j_p).orientation;
                            % Added a rule to avoide divergence of
                            % dislocation at the segment junctions where only
                            % the forces of these dislocations are forced
                            % to be zero since dislocation will tend to
                            % cluster at there. The rule is simply a flag
                            % that will be recalled at the get motion
                            % function
                               if (r_pq<C(i_q).B(k_q).mode(m_q).d/recomb_L || r_pq<C(i_p).B(k_p).mode(m_p).d/recomb_L) && annihilation==0
                                   if k_q==k_p
                                  C(i_q).B(k_q).mode(m_q).D(j_q).b=[0 0 0];
                                  C(i_p).B(k_p).mode(m_p).D(j_p).b=[0 0 0];
                                  C(i_q).B(k_q).mode(m_q).annihilation_flag=1;
                                   else
                                  C(i_q).B(k_q).mode(m_q).D(j_q).flag=1;
                                  C(i_p).B(k_p).mode(m_p).D(j_p).flag=1;
                                  [stress,displacement]=dislocation_field(R_pq*QQ);
                                  stress=norm(C(i_p).B(k_p).mode(m_p).D(j_p).b)*QQ*stress*QQ';
                                  C(i_q).B(k_q).mode(m_q).D(j_q).sigma= C(i_q).B(k_q).mode(m_q).D(j_q).sigma+stress;
                                   end

                               elseif r_pq==0 
                                      'self force=0';
                               else
                                [stress,displacement]=dislocation_field(R_pq*QQ);
                                stress=norm(C(i_p).B(k_p).mode(m_p).D(j_p).b)*QQ*stress*QQ';
                                C(i_q).B(k_q).mode(m_q).D(j_q).sigma= C(i_q).B(k_q).mode(m_q).D(j_q).sigma+stress;
                               end

                           C(i_q).B(k_q).mode(m_q).D(j_q).recomb=ceil(norm(C(i_q).B(k_q).mode(m_q).D(j_q).b));
                           C(i_p).B(k_p).mode(m_p).D(j_p).recomb=ceil(norm(C(i_p).B(k_p).mode(m_p).D(j_p).b));
                            
                      end
                  end
              end
          end

        [t,position]=get_motion(i_q,j_q,k_q,m_q);
        C(i_q).B(k_q).mode(m_q).t(j_q)=t;
        C(i_q).B(k_q).mode(m_q).position(j_q) = position;
        
        
        [x_cod_up, y_cod_up,x_cod_down,y_cod_down]=COD(i_q,j_q,k_q,m_q);
        C(i_q).B(k_q).mode(m_q).x_cod_up(j_q) = x_cod_up;
        C(i_q).B(k_q).mode(m_q).x_cod_down(j_q) = x_cod_down;
        C(i_q).B(k_q).mode(m_q).y_cod_up(j_q) = y_cod_up;
        C(i_q).B(k_q).mode(m_q).y_cod_down(j_q) = y_cod_down;

      end

      max_climb=max(abs(C(i_q).B(k_q).mode(m_q).t(2:(2*C(i_q).B(k_q).mode(m_q).n_d-1))));
      if C(i_q).B(k_q).mode(m_q).n_d==1
         C(i_q).B(k_q).mode(m_q).force_ratio=eps;
      else
         C(i_q).B(k_q).mode(m_q).force_ratio = max_climb/max(abs(C(i_q).B(k_q).mode(m_q).t(:)),[],"all");
         C(i_q).B(k_q).mode(m_q).PK_force = max(abs(C(i_q).B(k_q).mode(m_q).t(2:(2*C(i_q).B(k_q).mode(m_q).n_d-1)).*G.*Length_unit));
      end

      % The inner dislocations convergence criteria was changed to PK force
      % magnitude should be less that a very small value "eps"
         PKtemp = C(i_q).B(k_q).mode(m_q).PK_force;
         convergence = C(i_q).B(k_q).mode(m_q).PK_force;
         % plot_forces(i_q,k_q,m_q);   
end
plot_forces(i_q,k_q,m_q);           

% Calculating the stress intensity factor for each mode and each crack
% segment 
if C(i_q).B(k_q).case == "pure"
   C(i_q).B(k_q).mode(m_q).K = [sqrt(abs(C(i_q).B(k_q).mode(m_q).t(1))*Length_unit*G*Eeff),sqrt(abs(C(i_q).B(k_q).mode(m_q).t(2*C(i_q).B(k_q).mode(m_q).n_d))*Length_unit*G*Eeff)]
   K_th = sig_e(2,2)*G*sqrt(pi*C(i_q).B(k_q).a*Length_unit)
else
   C(i_q).B(k_q).mode(m_q).K = [sqrt(abs(C(i_q).B(k_q).mode(m_q).t(1))*Length_unit*G*Eeff),sqrt(abs(C(i_q).B(k_q).mode(m_q).t(2*C(i_q).B(k_q).mode(m_q).n_d))*Length_unit*G*Eeff)]
end

% Calculate the theoritical value ( valid only for uniaxial loading cases )
if m_q==1
       KI_th = C(i_q).B(k_q).sig_e_rot(2,2)*G*sqrt(pi*C(i_q).B(k_q).a*Length_unit) 
else
       KII_th = C(i_q).B(k_q).sig_e_rot(1,2)*G*sqrt(pi*C(i_q).B(k_q).a*Length_unit)
end

t_max = max(C(i_q).B(k_q).mode(m_q).t(:),[],"all");
                
for j=2:2*C(i_q).B(k_q).mode(m_q).n_d
    if norm( C(i_q).B(k_q).mode(m_q).D(j).b )==0
        C(i_q).B(k_q).mode(m_q).x_cod_up(j)=C(i_q).B(k_q).mode(m_q).x_cod_up(j-1);
        C(i_q).B(k_q).mode(m_q).y_cod_up(j)=C(i_q).B(k_q).mode(m_q).y_cod_up(j-1);
        C(i_q).B(k_q).mode(m_q).x_cod_down(j)=C(i_q).B(k_q).mode(m_q).x_cod_down(j-1);
        C(i_q).B(k_q).mode(m_q).y_cod_down(j)=C(i_q).B(k_q).mode(m_q).y_cod_down(j-1);
    end          
end


if C(i_q).B(k_q).mode(m_q).annihilation_flag==0 && C(i_q).B(k_q).mode(m_q).n_d<n_d_min

% Add two dipoles as long as there is no annihilation and the number of
% dislocations are less than a specified minimum number of dislocations

GetDipoles(i_q,k_q,m_q)

else

    % If annihilation occur before the specified number of dislocations
    % then reset the segments by deleting the last two dipoles and adding
    % two new one along with decreasing the burgers vector magnitude value
    % by a specified amount 

    if C(i_q).B(k_q).mode(m_q).n_d<n_d_min
      rset(i_q,k_q,m_q,1)
      C(i_q).B(k_q).mode(m_q).reset_flag=1;
    else

    % If annihilation did not occur reset the segment by deleting the last two dipoles and adding
    % two new one along with increasing the burgers vector magnitude value
    % by a specified amount ( Note: I will change this so it is related to
    % the maximum number of dislocations instead to the minumum specified
    % value )

      if C(i_q).B(k_q).mode(m_q).annihilation_flag==0 && rsc==0 
      rset(i_q,k_q,m_q,2)
      C(i_q).B(k_q).mode(m_q).reset_flag=1;
      rsc=1;
      end
    end
end

                end
            end
            % Calculate the effective stress intensity factor for the crack
            % segment 
            if C(i_q).B(k_q).case == "pure"
                C(i_q).B(k_q).K = C(i_q).B(k_q).mode(m_q).K;
            else
                C(i_q).B(k_q).K = sqrt(C(i_q).B(k_q).mode(1).K.^2+C(i_q).B(k_q).mode(2).K.^2);
            end
        end
     end
end
            % calculate_COD();
            % plot_COD();
            % plot_barriers();
end