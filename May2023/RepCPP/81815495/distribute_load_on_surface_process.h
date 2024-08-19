
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"

namespace Kratos {




class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) DistributeLoadOnSurfaceProcess 
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(DistributeLoadOnSurfaceProcess);

typedef std::size_t SizeType;


DistributeLoadOnSurfaceProcess(ModelPart& rModelPart,
Parameters Parameters);


void ExecuteInitializeSolutionStep() override;


virtual std::string Info() const override {
return "DistributeLoadOnSurfaceProcess";
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << "DistributeLoadOnSurfaceProcess";
}

void PrintData(std::ostream& rOStream) const override {
}


private:

ModelPart& mrModelPart; 
Parameters mParameters; 



}; 


}  