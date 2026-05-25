function Toughness=get_toughness(R,i)
global sig_amp r_b c_bx c_by


        tough_factor=100;
        T=tough_factor*sig_amp;
        a=1000;
        b=20000;
        c=1000;
        d=20000;
        Toughness=0;
        c_b=[c_bx(i) c_by(i) 0];
   distance=norm(R-c_b);

      
            if (distance<r_b(i))||((R(1)<a || R(1)>b || R(2)<c || R(2)>d))
                Toughness=T;
            end

end
