function update()


global  n_c  C  sig_amp n_branch 


for i=1:n_c
    for k=1:n_branch
        for m=1:C(i).B(k).n_modes

            C(i).B(k).mode(m).t=ones(1,2*C(i).B(k).mode(m).n_d)*sig_amp;
            C(i).B(k).mode(m).position=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).x_cod_up=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).y_cod_up=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).x_cod_down=zeros(1,2*C(i).B(k).mode(m).n_d);
            C(i).B(k).mode(m).y_cod_down=zeros(1,2*C(i).B(k).mode(m).n_d);


            C(i).B(k).mode(m).d=2*C(i).B(k).a/(2*C(i).B(k).mode(m).n_d-1);
            % C(i).B(k).mode(m).d=2*C(i).B(k).a/(2*2-1);
            C(i).B(k).mode(m).steps=max(50,C(i).B(k).mode(m).d/10);
            C(i).B(k).mode(m).iter_max=10*C(i).B(k).mode(m).steps;
        
            for j=1:2*C(i).B(k).mode(m).n_d       
              % C(i).B(k).D(j).sigma_disk=zeros(3,3);
                C(i).B(k).mode(m).D(j).sigma_uniaxial=zeros(3,3);                           %uniaxial
                C(i).B(k).mode(m).D(j).F_PK=ones(1,3);
                C(i).B(k).mode(m).D(j).b=-C(i).B(k).mode(m).b_mag*C(i).B(k).mode(m).b_direction;
                C(i).B(k).mode(m).D(j).recomb=1;
                if j<=C(i).B(k).mode(m).n_d
                   C(i).B(k).mode(m).D(j).orientation=+1;
                else
                   C(i).B(k).mode(m).D(j).orientation=-1;
                end
                   C(i).B(k).mode(m).D(j).flag=0; 
            end

        end

    end

end

end
