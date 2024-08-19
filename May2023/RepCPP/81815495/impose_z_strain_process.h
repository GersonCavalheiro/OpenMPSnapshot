
#pragma once

#include "processes/process.h"
#include "includes/model_part.h"

#include "structural_mechanics_application_variables.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ImposeZStrainProcess
: public Process
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ImposeZStrainProcess);


ImposeZStrainProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~ImposeZStrainProcess() override = default;


void operator()()
{
Execute();
}

void Execute() override;

void ExecuteInitializeSolutionStep() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "ImposeZStrainProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ImposeZStrainProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;


private:

ImposeZStrainProcess& operator=(ImposeZStrainProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
ImposeZStrainProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const ImposeZStrainProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
