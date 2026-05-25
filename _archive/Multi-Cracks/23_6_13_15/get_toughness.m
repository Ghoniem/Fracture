function Toughness=get_toughness(R,i)
global sig_amp r_b c_bx c_by


        tough_factor=100;
        T=tough_factor*sig_amp;
        a=6000;
        b=14000;
        c=3000;
        d=14000;
        Toughness=0;
        c_b=[c_bx(i) c_by(i) 0];
   distance=norm(R-c_b);

      
            if (distance<r_b(i))||((R(1)<a || R(1)>b || R(2)<c || R(2)>d))
                Toughness=T;
            end

end
