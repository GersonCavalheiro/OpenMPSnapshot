
#pragma once



#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SetAutomatedInitialVariableProcess
: public Process
{

public:

static constexpr double tolerance         = 1.0e-6;
static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(SetAutomatedInitialVariableProcess);


SetAutomatedInitialVariableProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~SetAutomatedInitialVariableProcess() override = default;


void ExecuteInitialize() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "SetAutomatedInitialVariableProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SetAutomatedInitialVariableProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;

private:

SetAutomatedInitialVariableProcess& operator=(SetAutomatedInitialVariableProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
SetAutomatedInitialVariableProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SetAutomatedInitialVariableProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 

