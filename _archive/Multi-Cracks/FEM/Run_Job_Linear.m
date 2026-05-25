function Job=Run_Job_Linear(Model,Loading_Steps)

NSteps=length(Loading_Steps);

for step=1:1:NSteps

Model=Analysis_Linear(Model,Loading_Steps{step}.Force,Loading_Steps{step}.U_s);

Job{step}.Model=Model;

end

end




