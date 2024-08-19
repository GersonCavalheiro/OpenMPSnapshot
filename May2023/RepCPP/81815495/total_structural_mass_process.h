
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) TotalStructuralMassProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(TotalStructuralMassProcess);


TotalStructuralMassProcess(
ModelPart& rThisModelPart
):mrThisModelPart(rThisModelPart)
{
KRATOS_TRY

KRATOS_CATCH("")
}

~TotalStructuralMassProcess() override
= default;







void Execute() override;

static double CalculateElementMass(Element& rElement, const std::size_t DomainSize);






std::string Info() const override
{
return "TotalStructuralMassProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "TotalStructuralMassProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart;              









TotalStructuralMassProcess& operator=(TotalStructuralMassProcess const& rOther) = delete;

TotalStructuralMassProcess(TotalStructuralMassProcess const& rOther) = delete;



}; 






}
