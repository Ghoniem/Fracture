function T = getTransformationMatrix(i,k)
    global C

    T = [cos(C(i).B(k).phi)  sin(C(i).B(k).phi) 0;
        -sin(C(i).B(k).phi)  cos(C(i).B(k).phi) 0;
        0                           0             1];

end
    
