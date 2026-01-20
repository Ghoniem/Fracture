function get_cracks()
global C n_c n_b r_b c_bx c_by X_intercept Y_intercept C_C_left C_C_right...
    a_crack D Length_unit
        
%%%% PREPARE CRACK & BARRIER DATA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X_intercept=zeros(n_c,n_c);
    Y_intercept=zeros(n_c,n_c);
    C_C_left=zeros(n_c,n_c);
    C_C_right=zeros(n_c,n_c);

    a1 = 500;
    a2 = 1000;
    R_c1=[5000 5000 0];
    R_c2=[15000 15000 0];
    phi1=0;
    phi2=pi;
    r_b1=0;
    r_b2=0;
    c_b1=[100 100 0];
    c_b2=[18000 18000 0];
    rng(0,'twister');
C(1).a=a_crack;
C(1).R_c=[0 0 0];
C(1).phi=pi/2;
C(2).a=a_crack/2;
C(2).R_c=[0.15*(D/(2*Length_unit)) (D/(3*Length_unit)) 0];
C(2).phi=pi/4;
C(3).a=500;
C(3).R_c=[8000 13000 0];
C(3).phi=2*pi/3;

%%%%%%%%%%%%%%%%%%%%%%%%%  CRACKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         for i=1:n_c
%             C(i).a=(a2-a1).*rand(1,1) + a1;
%             C(i).R_c=(R_c2-R_c1).*rand(1,1) + R_c1;
%             C(i).phi=(phi2-phi1).*rand(1,1) + phi1;
%         end

    for i=1:n_c
        for j=1:n_c
            if i~=j
    X_intercept(i,j)=(C(i).R_c(1)*tan(C(i).phi)-C(j).R_c(1)*tan(C(j).phi)-...
        (C(i).R_c(2)-C(j).R_c(2)))/(tan(C(i).phi)-tan(C(j).phi));
    Y_intercept(i,j)=C(i).R_c(2)+(X_intercept(i,j)-C(i).R_c(1))*tan(C(i).phi);
            end
        end
    end


%%%%%%%%%%%%%%%%%%%%%%%%%% BARRIERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:n_b
        r_b(i)=(r_b2-r_b1).*rand(1,1) + r_b1 ;   
        c_bx(i)=(c_b2(1)-c_b1(1)).*rand(1,1) + c_b1(1) ;  
        c_by(i)=(c_b2(2)-c_b1(2)).*rand(1,1) + c_b1(2) ;  
        end

        
%  plot_barriers();

end