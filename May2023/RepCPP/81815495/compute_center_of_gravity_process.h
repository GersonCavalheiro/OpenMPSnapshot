
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ComputeCenterOfGravityProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ComputeCenterOfGravityProcess);


ComputeCenterOfGravityProcess(
ModelPart& rThisModelPart
):mrThisModelPart(rThisModelPart)
{
KRATOS_TRY

KRATOS_CATCH("")
}

~ComputeCenterOfGravityProcess() override
= default;







void Execute() override;






std::string Info() const override
{
return "ComputeCenterOfGravityProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ComputeCenterOfGravityProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart;              









ComputeCenterOfGravityProcess& operator=(ComputeCenterOfGravityProcess const& rOther) = delete;

ComputeCenterOfGravityProcess(ComputeCenterOfGravityProcess const& rOther) = delete;



}; 






}
