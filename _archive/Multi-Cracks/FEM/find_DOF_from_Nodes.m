function [DOFx, DOFy, Node]=find_DOF_from_Nodes(Nodes,x,y)

DOFx=[];
DOFy=[];

NNode=size(Nodes,1);

if ~isempty(x) && ~isempty(y)

for i=1:1:NNode
    
if Nodes(i,1)==x && Nodes(i,2)==y

DOFx=2*i-1;
DOFy=2*i;
Node=i;
break;

end

end

end


%------------------------------------

if ~isempty(x) && isempty(y)

c=0;
for i=1:1:NNode
    
if Nodes(i,1)==x 
c=c+1;    
DOFx(c)=2*i-1;
DOFy(c)=2*i;
Node(c)=i;

end

end

end

%-------------------------------------

if isempty(x) && ~isempty(y)
    
c=0;
for i=1:1:NNode
  
if Nodes(i,2)==y
c=c+1;     
DOFx(c)=2*i-1;
DOFy(c)=2*i;
Node(c)=i;

end

end

end



end