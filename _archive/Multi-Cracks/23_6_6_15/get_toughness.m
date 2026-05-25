function Toughness=get_toughness(R)
global C sig_amp

T1=0.5*sig_amp;
T2=20*sig_amp;
a=9000;
b=11000;
c=8000;
d=12000;
if R(1)<a || R(1)>b || R(2)<c || R(2)>d
    Toughness=T2;
else 
    Toughness=T1;
end
