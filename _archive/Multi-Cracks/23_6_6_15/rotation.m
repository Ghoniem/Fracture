function QQ=rotation (i,j)
global C 
        if j<=C(i).n_d
%             C(i).D(j).rot=0+C(i).phi;
            C(i).D(j).rot=C(i).theta+C(i).phi;

          else
%             C(i).D(j).rot=-pi+C(i).phi;
            C(i).D(j).rot=-pi+C(i).theta+C(i).phi;

        end
            sn=sin(C(i).D(j).rot);
            cs=cos(C(i).D(j).rot);
            C(i).D(j).Q=[cs sn 0; -sn cs 0;0 0 1];
            QQ=C(i).D(j).Q;
end
