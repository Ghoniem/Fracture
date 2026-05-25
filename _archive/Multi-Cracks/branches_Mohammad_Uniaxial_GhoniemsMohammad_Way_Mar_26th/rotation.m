function QQ=rotation (i,j,k,m,flag)
global C 

switch(flag)

    case 0
        if j<=C(i).B(k).mode(m).n_d
            C(i).B(k).mode(m).D(j).rot=C(i).B(k).phi;

          else
            C(i).B(k).mode(m).D(j).rot=pi+C(i).B(k).phi;

        end


            sn=sin(C(i).B(k).mode(m).D(j).rot);
            cs=cos(C(i).B(k).mode(m).D(j).rot);
            C(i).B(k).mode(m).D(j).Q=[-sn -cs 0; cs -sn 0;0 0 1];
            QQ=C(i).B(k).mode(m).D(j).Q;

    case 1

   if j<=C(i).B(k).mode(m).n_d
            C(i).B(k).mode(m).D(j).rot=C(i).B(k).phi;

          else
            C(i).B(k).mode(m).D(j).rot=pi+C(i).B(k).phi;

        end


            sn=sin(C(i).B(k).mode(m).D(j).rot);
            cs=cos(C(i).B(k).mode(m).D(j).rot);
            C(i).B(k).mode(m).D(j).T=[cs -sn 0; sn cs 0;0 0 1];
            QQ=C(i).B(k).mode(m).D(j).T;

end
end
