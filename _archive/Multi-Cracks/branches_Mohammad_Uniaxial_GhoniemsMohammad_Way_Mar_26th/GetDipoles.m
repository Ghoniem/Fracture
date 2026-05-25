function GetDipoles(i_q,k_q,m_q)

global C

       position_temp=ones(2*C(i_q).B(k_q).mode(m_q).n_d,3);
   for j_s =1:2*(C(i_q).B(k_q).mode(m_q).n_d) 
       position_temp(j_s,:) = C(i_q).B(k_q).mode(m_q).D(j_s).R_p;
   end 

position_new=ones(length(position_temp(:,1))+2,3);
L=length(position_temp(:,1))+2;
position_new(1:L/2-1,:)=position_temp(1:length(position_temp(:,1))/2,:);
position_new(L/2+2:end,:)=position_temp(length(position_temp(:,1))/2+1:end,:);
position_new(L/2,:)=position_new(L/2-1,:)+C(i_q).B(k_q).mode(m_q).d*C(i_q).B(k_q).e_p;   % Seeded dipole
position_new(L/2+1,:)=position_new(L/2+2,:)-C(i_q).B(k_q).mode(m_q).d*C(i_q).B(k_q).e_p;  % Seeded dipole
    
C(i_q).B(k_q).mode(m_q).n_d=C(i_q).B(k_q).mode(m_q).n_d+1;

for j_s =1:2*(C(i_q).B(k_q).mode(m_q).n_d) 
    C(i_q).B(k_q).mode(m_q).D(j_s).R_p = position_new(j_s,:);
end