function Toughness=get_toughness(R)
global sig_amp R_min R_max theta_min theta_max n_b
tough_factor=100;

T=tough_factor*sig_amp;
a=1000;
b=19000;
c=1000;
d=19000;

%     if R(1)<a || R(1)>b || R(2)<c || R(2)>d
%         Toughness=T;
%     else 
%         Toughness=0;
%     end
theta=atan2(R(2),R(1));
        for i=1:n_b
            if ((R_min(i,1) <norm(R)) && (norm(R)<R_max(i,1)) && (theta_min(i,1)<theta) && (theta<theta_max(i,1)))...
                    ||(  R(1)<a || R(1)>b || R(2)<c || R(2)>d)
                Toughness=T;
            else
                Toughness=0;
            end
        end

end
