function Toughness=get_toughness(R,i)
global sig_amp r_b c_bx c_by n_grid del_grid


        tough_factor=100;
        T=tough_factor*sig_amp;
        tough_limit=n_grid*del_grid/2;
        a=-0.5*tough_limit;
        b=0.5*tough_limit;
        c=-0.5*tough_limit;
        d=0.5*tough_limit;
        Toughness=0;
        c_b=[c_bx(i) c_by(i) 0];
   distance=norm(R-c_b);

      
            if (distance<r_b(i))||((R(1)<a || R(1)>b || R(2)<c || R(2)>d))
                Toughness=T;
            end

end
