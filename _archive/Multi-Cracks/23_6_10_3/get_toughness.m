function Toughness=get_toughness(R)
global C sig_amp tough_factor

T=tough_factor*sig_amp;
a=8000;
b=14000;
c=8000;
d=12000;
if R(1)<a || R(1)>b || R(2)<c || R(2)>d
    Toughness=T;
else 
    Toughness=0;
end
