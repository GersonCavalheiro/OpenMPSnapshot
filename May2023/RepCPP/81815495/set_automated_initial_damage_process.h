
#pragma once



#include "processes/process.h"

namespace Kratos
{



class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) SetAutomatedInitialDamageProcess
: public Process
{

public:

static constexpr double tolerance         = 1.0e-6;
static constexpr double machine_tolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION(SetAutomatedInitialDamageProcess);


SetAutomatedInitialDamageProcess(
ModelPart& rThisModelPart,
Parameters ThisParameters = Parameters(R"({})")
);


~SetAutomatedInitialDamageProcess() override = default;


void ExecuteInitializeSolutionStep() override;


const Parameters GetDefaultParameters() const override;

std::string Info() const override
{
return "SetAutomatedInitialDamageProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "SetAutomatedInitialDamageProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:


ModelPart& mrThisModelPart;
Parameters mThisParameters;

private:

SetAutomatedInitialDamageProcess& operator=(SetAutomatedInitialDamageProcess const& rOther);


}; 

inline std::istream& operator >> (std::istream& rIStream,
SetAutomatedInitialDamageProcess& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const SetAutomatedInitialDamageProcess& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 