function rset(i,k,m,flag)

global  C

position_temp=ones(2*C(i).B(k).mode(m).n_d,3);

   for j_s =1:2*(C(i).B(k).mode(m).n_d) 
       position_temp(j_s,:) = C(i).B(k).mode(m).D(j_s).R_p;
   end

position_new=ones(length(position_temp(:,1))-2,3);
L=length(position_temp(:,1))-2;
position_new(1:L/2,:)=position_temp(1:length(position_temp(:,1))/2-1,:);
position_new(L/2+1:end,:)=position_temp(length(position_temp(:,1))/2+2:end,:);
C(i).B(k).mode(m).n_d=C(i).B(k).mode(m).n_d-1;

   for j_s =1:2*(C(i).B(k).mode(m).n_d) 
       C(i).B(k).mode(m).D(j_s).R_p = position_new(j_s,:);
   end

position_temp=ones(2*C(i).B(k).mode(m).n_d,3);

   for j_s =1:2*(C(i).B(k).mode(m).n_d) 
       position_temp(j_s,:) = C(i).B(k).mode(m).D(j_s).R_p;
   end 

position_new=ones(length(position_temp(:,1))+2,3);
L=length(position_temp(:,1))+2;
position_new(1:L/2-1,:)=position_temp(1:length(position_temp(:,1))/2,:);
position_new(L/2+2:end,:)=position_temp(length(position_temp(:,1))/2+1:end,:);
position_new(L/2,:)=position_new(L/2-1,:)+C(i).B(k).mode(m).d*C(i).B(k).e_p;   % Seeded dipole
position_new(L/2+1,:)=position_new(L/2+2,:)-C(i).B(k).mode(m).d*C(i).B(k).e_p;  % Seeded dipole
    
C(i).B(k).mode(m).n_d=C(i).B(k).mode(m).n_d+1;

    for j_s =1:2*(C(i).B(k).mode(m).n_d) 
        C(i).B(k).mode(m).D(j_s).R_p = position_new(j_s,:);
    end

if flag==1
   C(i).B(k).mode(m).b_mag=0.8*C(i).B(k).mode(m).b_mag; 
else
   C(i).B(k).mode(m).b_mag=1.2*C(i).B(k).mode(m).b_mag; 
end

    for j=1:2*C(i).B(k).mode(m).n_d 
        C(i).B(k).mode(m).D(j).b=-C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction;
        if j<=C(i).B(k).mode(m).n_d
           C(i).B(k).mode(m).D(j).orientation=+1;
        else
             C(i).B(k).mode(m).D(j).orientation=-1;
        end

        C(i).B(k).mode(m).D(j).flag=0; 
    end

C(i).B(k).mode(m).force_ratio=ones(1,1);
C(i).B(k).mode(m).PK_force=ones(1,1);
C(i).B(k).mode(m).annihilation_flag=0;

end
