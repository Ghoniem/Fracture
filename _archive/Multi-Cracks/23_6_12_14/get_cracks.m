function get_cracks()
global C n_c n_b r_b c_bx c_by
        
%%%% PREPARE CRACK & BARRIER DATA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a1 = 500;
    a2 = 1000;
    R_c1=[5000 5000 0];
    R_c2=[15000 15000 0];
    phi1=0;
    phi2=pi;
    r_b1=100;
    r_b2=400;
    c_b1=[100 100 0];
    c_b2=[18000 18000 0];
    rng(0,'twister');


        for i=1:n_c
            C(i).a=(a2-a1).*rand(1,1) + a1;
            C(i).R_c=(R_c2-R_c1).*rand(1,1) + R_c1;
            C(i).phi=(phi2-phi1).*rand(1,1) + phi1;
        end

        for i=1:n_b
        r_b(i)=(r_b2-r_b1).*rand(1,1) + r_b1 ;   
        c_bx(i)=(c_b2(1)-c_b1(1)).*rand(1,1) + c_b1(1) ;  
        c_by(i)=(c_b2(2)-c_b1(2)).*rand(1,1) + c_b1(2) ;  
        end

        
 plot_barriers();

end