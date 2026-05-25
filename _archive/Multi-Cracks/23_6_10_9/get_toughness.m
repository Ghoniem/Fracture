function Toughness=get_toughness(R)
global C sig_amp tough_factor

T=tough_factor*sig_amp;
a=9000;
b=13000;
c=13000;
d=19000;
if R(1)<a || R(1)>b || R(2)<c || R(2)>d
    Toughness=T;
else 
    Toughness=0;
end
